"""
joint_analog.py

Joint Analog Simulation

The Transformer encoder is trained solely on the wavelet-reconstructed signal
(identical to analog_mode in HybridKNNTransformer). After training, a joint
store is built that maps each training embedding to the full observed tuple
(signal, frequency, max_intensity, duration_days) across all sites for the
corresponding pred_horizon window.

At forecast time, the encoder embeds the current climate state and retrieves
K nearest historic windows by embedding distance. Flood characteristics are
drawn jointly from these analog windows — preserving the full spatial
dependence structure across sites by construction.

Key design properties:
  - No changes to Transformer training
  - No copula fitting or marginal specification required
  - Spatial joint dependence in flood characteristics inherited from historic record
  - Zero-event months included in pool (preserves intermittency)
  - Output is a drop-in replacement for (next_ens, resampled_all) in the notebook

Integration:
  - Requires a trained HybridKNNTransformer (analog_mode or forecast_mode)
  - Requires climate_filtered DataFrame with both signal and flood characteristic columns
  - Output format matches existing resampled_all / next_ens conventions
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch

from transformer_knn import (
    HybridKNNTransformer,
    KNNMeanConfig,
    knn_topk,
    softmax_weights,
    _to_tensor,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class JointAnalogConfig:
    """
    Configuration for JointAnalogSampler.

    Parameters
    ----------
    K : int
        Number of nearest neighbors in embedding space. Defaults to
        sqrt(n_train_sequences), clipped to [5, 200]. Can be set explicitly.
    temperature : float
        Softmax temperature for neighbor sampling weights.
        Higher = more uniform; lower = concentrate on nearest neighbor.
    flood_vars : list[str]
        Suffixes of flood characteristic columns to retrieve, e.g.
        ["_frequency", "_max_intensity", "_duration_days"].
        Full column names are constructed as {site}{suffix}.
    signal_suffix : str
        Suffix used to identify signal columns, e.g. "_max_z_wave".
        Full column names are constructed as {site}{signal_suffix}.
    seed : int or None
        Random seed for reproducible ensemble draws. None = random each call.
    """
    K: int = 20
    temperature: float = 1.0
    flood_vars: List[str] = None
    signal_suffix: str = "_max_z_wave"
    seed: Optional[int] = 0

    def __post_init__(self):
        if self.flood_vars is None:
            self.flood_vars = ["_frequency", "_max_intensity", "_duration_days"]


# ---------------------------------------------------------------------------
# Joint store entry
# ---------------------------------------------------------------------------

# Each entry: (embedding [H], joint_payload [pred_horizon, n_joint_cols])
# joint_payload columns = signal cols + flood char cols, ordered as:
#   [site0_signal, site1_signal, ..., site0_freq, site0_inten, site0_dur,
#    site1_freq, site1_inten, site1_dur, ...]
# Column ordering is stored in JointAnalogSampler.joint_cols for transparency.


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class JointAnalogSampler:
    """
    Joint analog sampler (Approach D).

    Usage
    -----
    # 1) Train transformer as usual (analog_mode or forecast_mode - doesn't matter)
    exp = HybridKNNTransformer(..., mode="analog_mode")
    exp.fit(climate_filtered, ...)

    # 2) Build joint store (replaces / extends mean_store)
    sampler = JointAnalogSampler(exp, sites, cfg=JointAnalogConfig(K=30))
    sampler.build_joint_store(climate_filtered)

    # 3) Sample ensemble (replaces next_ens + KNNConditionalCopulaSampler)
    resampled_all = sampler.sample_ensemble(climate_filtered, n_samples=1000)

    # resampled_all is a DataFrame with MultiIndex (sim_id, t) and columns:
    #   {site}_max_z_wave, {site}_frequency, {site}_max_intensity,
    #   {site}_duration_days, {site}_forecast_signal  for all sites
    # This is a drop-in replacement for resampled_all in the notebook.
    """

    def __init__(
        self,
        transformer: HybridKNNTransformer,
        sites: List[str],
        cfg: Optional[JointAnalogConfig] = None,
    ):
        self.transformer = transformer
        self.sites = list(sites)
        self.cfg = cfg or JointAnalogConfig()

        # populated by build_joint_store()
        self.joint_store: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self._joint_keys: Optional[torch.Tensor] = None
        self.joint_cols: List[str] = []   # ordered column names of joint_payload
        self._n_signal: int = 0           # number of signal cols (first block)
        self._n_flood: int = 0            # number of flood char cols (second block)

    # ------------------------------------------------------------------
    # Store construction
    # ------------------------------------------------------------------

    @torch.no_grad()
    def build_joint_store(self, df: pd.DataFrame) -> "JointAnalogSampler":
        """
        Build the joint store from df (typically climate_filtered).

        For each valid training sequence window, stores:
          - Transformer encoder embedding of the input window
          - Joint payload: signal + flood characteristics across all sites
            for the corresponding pred_horizon future window

        Parameters
        ----------
        df : pd.DataFrame
            Must contain all feature_columns used by the transformer and
            all joint columns ({site}{signal_suffix} and {site}{flood_var}
            for all sites and flood_vars in cfg).
        """
        exp = self.transformer
        cfg = self.cfg

        # --- resolve joint column names and validate ---
        signal_cols = [f"{s}{cfg.signal_suffix}" for s in self.sites]
        flood_cols = [
            f"{s}{fv}"
            for s in self.sites
            for fv in cfg.flood_vars
        ]
        self.joint_cols = signal_cols + flood_cols
        self._n_signal = len(signal_cols)
        self._n_flood = len(flood_cols)

        missing = [c for c in self.joint_cols if c not in df.columns]
        if missing:
            raise KeyError(
                f"Joint store columns missing from df: {missing}\n"
                f"Ensure flood characteristic columns have been joined to "
                f"climate_filtered before calling build_joint_store()."
            )

        # --- build input sequences (X) using transformer's feature cols ---
        # We reuse _make_xy but need the joint payload as y, not the signal targets.
        # So we construct sequences manually over the joint cols.
        from transformer_knn import create_sequences_np

        seq_len = exp.seq_length
        horizon = exp.pred_horizon

        # Drop rows with NaN in feature columns (same as transformer training)
        df_clean = df.dropna(subset=exp.feature_columns).copy()

        # Build X from feature columns
        X_np, _ = create_sequences_np(
            df_clean,
            feature_columns=exp.feature_columns,
            target_columns=exp.feature_columns[:1],  # dummy target, unused
            seq_length=seq_len,
            pred_horizon=horizon,
        )

        # Build joint payload manually: [N, horizon, n_joint_cols]
        joint_np = self._build_joint_payload_np(df_clean, seq_len, horizon)

        n_windows = min(len(X_np), len(joint_np))
        if n_windows == 0:
            raise RuntimeError(
                "No valid windows found. Check that df has enough rows "
                f"(need > seq_length + pred_horizon = {seq_len + horizon})."
            )

        X_np = X_np[:n_windows]
        joint_np = joint_np[:n_windows]

        # --- auto-set K if using default ---
        if self.cfg.K == 20:  # default sentinel
            k_auto = int(np.clip(round(np.sqrt(n_windows)), 5, 200))
            self.cfg.K = k_auto

        # --- encode all windows in batches ---
        exp.model.eval()
        device = exp.device
        X_t = _to_tensor(X_np, device)

        self.joint_store = []
        batch_size = 256
        for start in range(0, n_windows, batch_size):
            xb = X_t[start : start + batch_size]
            keys = exp.model.encoder(xb).detach().cpu()  # [B, H]
            for i in range(keys.shape[0]):
                payload = torch.tensor(
                    joint_np[start + i], dtype=torch.float32
                )  # [horizon, n_joint_cols]
                self.joint_store.append((keys[i], payload))

        self._joint_keys = torch.stack(
            [k for k, _ in self.joint_store], dim=0
        )  # [N, H]

        print(
            f"Joint store built: {len(self.joint_store)} windows, "
            f"K={self.cfg.K}, "
            f"joint_cols={len(self.joint_cols)} "
            f"({self._n_signal} signal + {self._n_flood} flood)"
        )
        return self

    def _build_joint_payload_np(
        self,
        df: pd.DataFrame,
        seq_len: int,
        horizon: int,
    ) -> np.ndarray:
        """
        Extract [N, horizon, n_joint_cols] array of future joint values.

        Window i covers rows [i + seq_len : i + seq_len + horizon].
        Mirrors the y-extraction logic in create_sequences_np.
        """
        joint_arr = df[self.joint_cols].to_numpy(dtype=np.float32)
        n = len(df)
        last_start = n - seq_len - horizon
        if last_start < 0:
            return np.zeros((0, horizon, len(self.joint_cols)), dtype=np.float32)

        payloads = np.stack(
            [
                joint_arr[i + seq_len : i + seq_len + horizon]
                for i in range(last_start + 1)
            ],
            axis=0,
        )  # [N, horizon, n_joint_cols]
        return payloads

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample_ensemble(
        self,
        df: pd.DataFrame,
        n_samples: int = 1000,
        return_signal_ensemble: bool = True,
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, np.ndarray]]:
        """
        Generate ensemble of joint (signal + flood characteristics) trajectories
        from the latest available state in df.

        Parameters
        ----------
        df : pd.DataFrame
            Full climate_filtered DataFrame. The last seq_length rows are used
            as the current state for embedding.
        n_samples : int
            Ensemble size.
        return_signal_ensemble : bool
            If True, also return next_ens array [n_samples, pred_horizon, n_sites]
            containing signal values, matching the existing next_ens convention
            used by the notebook for compatibility with Approaches A/B. This is
            the signal ensemble consumed by the copula conditioning step.

        Returns
        -------
        resampled_all : pd.DataFrame
            MultiIndex (sim_id, t), columns = joint_cols + {site}_forecast_signal.
            Drop-in replacement for resampled_all in the notebook.
        next_ens : np.ndarray, shape [n_samples, pred_horizon, n_sites]
            Signal ensemble only. Returned if return_signal_ensemble=True.
            Matches existing next_ens shape convention.
        """
        if not self.joint_store or self._joint_keys is None:
            raise RuntimeError("Call build_joint_store() before sample_ensemble().")

        exp = self.transformer
        cfg = self.cfg
        rng = np.random.default_rng(None if cfg.seed is None else int(cfg.seed))

        # --- embed current state ---
        if len(df) < exp.seq_length:
            raise ValueError(
                f"df has {len(df)} rows but seq_length={exp.seq_length}. "
                "Need at least seq_length rows."
            )

        window = df.iloc[-exp.seq_length:][exp.feature_columns].to_numpy(dtype=np.float32)
        x = _to_tensor(window[None, :, :], exp.device)  # [1, L, F]

        exp.model.eval()
        qkey = exp.model.encoder(x).squeeze(0).detach().cpu()  # [H]

        # --- KNN lookup ---
        K_eff = min(cfg.K, len(self.joint_store))
        topk_idx, topk_d = knn_topk(qkey, self._joint_keys, K_eff)
        weights = softmax_weights(topk_d, cfg.temperature).numpy()  # [K]

        # --- sample n_samples analog trajectories ---
        # Each draw picks one neighbor and retrieves its joint payload
        picks = rng.choice(np.arange(K_eff), size=n_samples, p=weights)

        # payloads: [n_samples, pred_horizon, n_joint_cols]
        payloads = np.stack(
            [self.joint_store[int(topk_idx[p])][1].numpy() for p in picks],
            axis=0,
        ).astype(np.float32)

        # --- assemble resampled_all DataFrame ---
        resampled_all = self._payloads_to_dataframe(payloads)

        if not return_signal_ensemble:
            return resampled_all

        # --- extract next_ens for the copula conditioning step ---
        # Shape: [n_samples, pred_horizon, n_sites]
        signal_col_indices = list(range(self._n_signal))
        next_ens = payloads[:, :, signal_col_indices]  # [E, H, S]

        return resampled_all, next_ens

    @torch.no_grad()
    def sample_ensemble_windowed(
        self,
        df: pd.DataFrame,
        n_samples: int = 1000,
    ) -> pd.DataFrame:
        """
        Generate ensemble over all available windows in df (analogous to
        predict_windows_ensemble). Useful for in-sample diagnostics and
        validation of joint dependence structure.

        Returns
        -------
        pd.DataFrame
            MultiIndex (sim_id, window_id, t), columns = joint_cols.
        """
        if not self.joint_store or self._joint_keys is None:
            raise RuntimeError("Call build_joint_store() before sample_ensemble_windowed().")

        from transformer_knn import create_sequences_np

        exp = self.transformer
        cfg = self.cfg
        rng = np.random.default_rng(None if cfg.seed is None else int(cfg.seed))

        df_clean = df.dropna(subset=exp.feature_columns).copy()
        X_np, _ = create_sequences_np(
            df_clean,
            feature_columns=exp.feature_columns,
            target_columns=exp.feature_columns[:1],
            seq_length=exp.seq_length,
            pred_horizon=exp.pred_horizon,
        )
        if len(X_np) == 0:
            raise ValueError("No valid windows in df for windowed sampling.")

        exp.model.eval()
        device = exp.device
        X_t = _to_tensor(X_np, device)

        K_eff = min(cfg.K, len(self.joint_store))

        all_frames = []
        batch_size = 256

        for start in range(0, len(X_t), batch_size):
            xb = X_t[start : start + batch_size]
            keys = exp.model.encoder(xb).detach().cpu()  # [B, H]

            for b in range(keys.shape[0]):
                win_id = start + b
                qkey = keys[b]
                topk_idx, topk_d = knn_topk(qkey, self._joint_keys, K_eff)
                weights = softmax_weights(topk_d, cfg.temperature).numpy()

                picks = rng.choice(np.arange(K_eff), size=n_samples, p=weights)
                payloads = np.stack(
                    [self.joint_store[int(topk_idx[p])][1].numpy() for p in picks],
                    axis=0,
                ).astype(np.float32)  # [E, H, n_joint_cols]

                frame = self._payloads_to_dataframe(payloads, window_id=win_id)
                all_frames.append(frame)

        return pd.concat(all_frames).sort_index()

    # ------------------------------------------------------------------
    # Output formatting
    # ------------------------------------------------------------------

    def _payloads_to_dataframe(
        self,
        payloads: np.ndarray,
        window_id: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Convert [n_samples, pred_horizon, n_joint_cols] array to a DataFrame
        with MultiIndex (sim_id, t) matching resampled_all convention.

        Also adds {site}_forecast_signal columns (alias of signal cols)
        for backward compatibility with downstream notebook cells.
        """
        n_samples, horizon, n_cols = payloads.shape

        sim_ids = np.repeat(np.arange(n_samples), horizon)
        t_vals = np.tile(np.arange(horizon), n_samples)

        flat = payloads.reshape(-1, n_cols)
        df = pd.DataFrame(flat, columns=self.joint_cols)

        # Add forecast_signal alias columns (backward compat with resampled_all)
        signal_cols = self.joint_cols[: self._n_signal]
        for site, sc in zip(self.sites, signal_cols):
            df[f"{site}_forecast_signal"] = df[sc].values

        if window_id is not None:
            df.index = pd.MultiIndex.from_arrays(
                [sim_ids, t_vals, np.full(len(df), window_id, dtype=int)],
                names=["sim_id", "t", "window_id"],
            )
        else:
            df.index = pd.MultiIndex.from_arrays(
                [sim_ids, t_vals],
                names=["sim_id", "t"],
            )

        return df

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def spatial_correlation_summary(
        self,
        resampled_all: pd.DataFrame,
        climate_filtered: pd.DataFrame,
        var_suffix: str = "_max_intensity",
        quantile_clip: Tuple[float, float] = (0.001, 0.999),
        min_pairs: int = 5,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Compare pairwise Pearson correlation of a flood characteristic
        between observed (climate_filtered) and simulated (resampled_all).

        Returns
        -------
        obs_corr : pd.DataFrame
            Observed pairwise correlation matrix.
        sim_corr : pd.DataFrame
            Simulated pairwise correlation matrix.
        """
        cols = [f"{s}{var_suffix}" for s in self.sites]

        obs_cols = [c for c in cols if c in climate_filtered.columns]
        sim_cols = [c for c in cols if c in resampled_all.columns]
        shared = [c for c in obs_cols if c in sim_cols]

        if not shared:
            raise ValueError(
                f"No shared columns found for var_suffix='{var_suffix}'. "
                f"Check that flood characteristic columns exist in both DataFrames."
            )

        obs_df = climate_filtered[shared].copy()
        sim_df = resampled_all.reset_index(level="sim_id", drop=True)[shared].copy()

        obs_corr = obs_df.corr(method="pearson", min_periods=min_pairs)
        sim_corr = sim_df.corr(method="pearson", min_periods=min_pairs)

        return obs_corr, sim_corr

    def tail_correlation_summary(
        self,
        resampled_all: pd.DataFrame,
        climate_filtered: pd.DataFrame,
        var_suffix: str = "_max_intensity",
        q: float = 0.9,
        min_pairs: int = 5,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Compare upper-tail pairwise correlation (both variables above q-quantile)
        between observed and simulated.
        """
        cols = [f"{s}{var_suffix}" for s in self.sites]
        shared = [c for c in cols if c in climate_filtered.columns and c in resampled_all.columns]

        if not shared:
            raise ValueError(f"No shared columns for var_suffix='{var_suffix}'.")

        def _tail_corr(df: pd.DataFrame) -> pd.DataFrame:
            n = len(shared)
            mat = np.full((n, n), np.nan)
            arr = df[shared].to_numpy(dtype=float)
            for i in range(n):
                for j in range(i, n):
                    xi, xj = arr[:, i], arr[:, j]
                    mask = np.isfinite(xi) & np.isfinite(xj)
                    if mask.sum() < min_pairs:
                        continue
                    thr_i = np.quantile(xi[mask], q)
                    thr_j = np.quantile(xj[mask], q)
                    tail = mask & (xi >= thr_i) & (xj >= thr_j)
                    if tail.sum() < min_pairs:
                        continue
                    c = np.corrcoef(xi[tail], xj[tail])[0, 1]
                    mat[i, j] = mat[j, i] = c
            return pd.DataFrame(mat, index=shared, columns=shared)

        sim_df = resampled_all.reset_index(level="sim_id", drop=True)[shared].copy()
        return _tail_corr(climate_filtered), _tail_corr(sim_df)
