"""
analog_quantile_sampler.py

Analog-Quantile Matched Copula Sampler
=======================================

Algorithm
---------
Two-step process combining parametric out-of-sample generation with
climate-state-conditioned joint structure:

Univariate quantile matching:
    For each (sim_id, t, site, variable), the joint analog payload value
    is mapped through the fitted marginal CDF to get u_analog in [0,1].
    The copula pool for that site/variable is pre-sorted by its CDF values.
    The copula value whose CDF position is nearest to u_analog is selected
    via searchsorted — O(log n) per query, fully vectorised.

    This ensures the selected values are at the right marginal level for
    the current climate state (high-signal analog months select from the
    high end of each variable's distribution) while preserving parametric
    tail extrapolation beyond the historical record.

Key properties
--------------
- Univariate marginals  : preserved (values from fitted copula pool)
- Tail extrapolation    : preserved (parametric marginals)
- Marginal level        : climate-conditioned via analog CDF position
- Out-of-sample         : achieved via copula parametric marginals
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


class AnalogQuantileSampler:
    """
    Univariate analog-quantile matched copula sampler.

    Parameters
    ----------
    copula_sims : Dict[str, pd.DataFrame]
        Per-site copula simulation tables (EmpiricalCopulaSimulator.sample()).

    copula_simulators : Dict[str, Any]
        Fitted EmpiricalCopulaSimulator objects. Must have called fit_marginals().

    sites : List[str]
        Site names in pipeline order.

    primary_vars : List[str]
        Variable suffixes in copula_sims, e.g. ["_frequency", "_max_z_wave"].
        Signal suffix is excluded from matching and written from next_ens.

    multi_copula_sims : Dict[str, pd.DataFrame], optional
        Per-site multi-event copula tables.

    multi_copula_simulators : Dict[str, Any], optional
        Fitted multi-event simulator objects.

    multi_vars : List[str], optional
        Variable suffixes in multi_copula_sims.
        Defaults to ["_duration_days", "_max_intensity", "_max_z_wave"].

    signal_suffix : str
        Suffix of signal column. Default "_max_z_wave".
        Excluded from quantile matching; overwritten from next_ens.

    jitter : float
        Std dev of Gaussian jitter on u_analog before searchsorted.
        Adds stochasticity across ensemble members sharing the same
        analog window. Default 0.02. Set 0.0 for deterministic output.

    seed : int or None
        Random seed for jitter.
    """

    def __init__(
        self,
        copula_sims: Dict[str, pd.DataFrame],
        copula_simulators: Dict[str, Any],
        sites: List[str],
        primary_vars: List[str],
        multi_copula_sims: Optional[Dict[str, pd.DataFrame]] = None,
        multi_copula_simulators: Optional[Dict[str, Any]] = None,
        multi_vars: Optional[List[str]] = None,
        signal_suffix: str = "_max_z_wave",
        jitter: float = 0.02,
        seed: Optional[int] = None,
    ):
        self.copula_sims             = copula_sims
        self.copula_simulators       = copula_simulators
        self.sites                   = list(sites)
        self.primary_vars            = list(primary_vars)
        self.multi_copula_sims       = multi_copula_sims or {}
        self.multi_copula_simulators = multi_copula_simulators or {}
        self.multi_vars              = list(multi_vars) if multi_vars is not None else [
            "_duration_days", "_max_intensity", "_max_z_wave"
        ]
        self.signal_suffix = signal_suffix
        self.jitter        = jitter
        self.rng           = np.random.default_rng(seed)

        # _sorted_u[site][col]   : (n_pool,) pool CDF values, sorted ascending
        # _sorted_raw[site][col] : (n_pool,) raw pool values sorted by CDF
        self._sorted_u:   Dict[str, Dict[str, np.ndarray]] = {}
        self._sorted_raw: Dict[str, Dict[str, np.ndarray]] = {}

        self._sorted_u_multi:   Dict[str, Dict[str, np.ndarray]] = {}
        self._sorted_raw_multi: Dict[str, Dict[str, np.ndarray]] = {}

        self._build_pools(primary=True)
        if self.multi_copula_sims:
            self._build_pools(primary=False)

    # ------------------------------------------------------------------
    # Pool construction
    # ------------------------------------------------------------------

    def _map_to_u(self, raw: np.ndarray, col: str, sim: Any) -> np.ndarray:
        """Map raw values to [0,1] via fitted marginal CDF or empirical rank."""
        if col in sim.marginals_:
            u = sim.marginals_[col].cdf(raw)
        else:
            n = len(raw)
            u = (np.argsort(np.argsort(raw)) + 1.0) / (n + 1.0)
        return np.clip(u.astype(float), 1e-6, 1 - 1e-6)

    def _build_pools(self, primary: bool = True) -> None:
        """Pre-sort each variable's copula pool by CDF value for searchsorted."""
        sims       = self.copula_sims       if primary else self.multi_copula_sims
        simulators = self.copula_simulators if primary else self.multi_copula_simulators
        vars_      = self.primary_vars      if primary else self.multi_vars
        su_d       = self._sorted_u         if primary else self._sorted_u_multi
        sr_d       = self._sorted_raw       if primary else self._sorted_raw_multi

        for site in self.sites:
            if site not in sims or site not in simulators:
                continue
            df, sim   = sims[site], simulators[site]
            su_d[site] = {}
            sr_d[site] = {}

            for v in vars_:
                col = f"{site}{v}"
                if v == self.signal_suffix or col not in df.columns:
                    continue
                raw   = df[col].to_numpy(dtype=float)
                u     = self._map_to_u(raw, col, sim)
                order = np.argsort(u)
                su_d[site][col] = u[order]
                sr_d[site][col] = raw[order]

    # ------------------------------------------------------------------
    # Core 1D match
    # ------------------------------------------------------------------

    def _match_u(
        self,
        u_queries: np.ndarray,  # (N,)
        sorted_u: np.ndarray,   # (n_pool,) sorted
        sorted_raw: np.ndarray, # (n_pool,) sorted by u
    ) -> np.ndarray:
        """Return pool raw values whose CDF positions are nearest to u_queries."""
        idx      = np.searchsorted(sorted_u, u_queries, side="left")
        idx      = np.clip(idx, 0, len(sorted_u) - 1)
        idx_left = np.maximum(idx - 1, 0)
        use_left = np.abs(sorted_u[idx_left] - u_queries) < np.abs(sorted_u[idx] - u_queries)
        best     = np.where(use_left, idx_left, idx)
        return sorted_raw[best]

    def _analog_to_u(self, raw: np.ndarray, col: str, sim: Any) -> np.ndarray:
        """Map analog raw values to [0,1] via fitted marginal CDF."""
        if col in sim.marginals_:
            u = sim.marginals_[col].cdf(raw)
        else:
            n = len(raw)
            u = (np.argsort(np.argsort(raw)) + 1.0) / (n + 1.0)
        return np.clip(u.astype(float), 1e-6, 1 - 1e-6)

    # ------------------------------------------------------------------
    # Primary sample
    # ------------------------------------------------------------------

    def sample(
        self,
        analog_payloads: np.ndarray,
        next_ens: np.ndarray,
        analog_joint_cols: List[str],
    ) -> pd.DataFrame:
        """
        Build resampled_all via univariate quantile matching.

        Call apply_schaake_shuffle(flood_vars=["_frequency"]) after this
        to restore cross-site and temporal joint structure.

        Parameters
        ----------
        analog_payloads : np.ndarray, shape (E, T, n_joint_cols)
        next_ens        : np.ndarray, shape (E, T, S)
        analog_joint_cols : List[str]  — JointAnalogSampler.joint_cols

        Returns
        -------
        pd.DataFrame, MultiIndex (sim_id, t)
        """
        E, T, _ = analog_payloads.shape
        S       = len(self.sites)
        assert next_ens.shape == (E, T, S), (
            f"next_ens shape {next_ens.shape} != ({E}, {T}, {S})"
        )

        ac_idx      = {col: i for i, col in enumerate(analog_joint_cols)}
        all_sim_ids = np.repeat(np.arange(E), T)
        all_t_idx   = np.tile(np.arange(T), E)
        out_arrays: Dict[str, np.ndarray] = {}

        for site_j, site in enumerate(self.sites):
            if site not in self._sorted_u:
                continue
            sim = self.copula_simulators[site]

            for v in self.primary_vars:
                col = f"{site}{v}"
                if v == self.signal_suffix or col not in self._sorted_u[site]:
                    continue

                su = self._sorted_u[site][col]
                sr = self._sorted_raw[site][col]

                if col not in ac_idx:
                    out_arrays[col] = np.full(E * T, np.median(sr))
                    continue

                raw_analog = analog_payloads[all_sim_ids, all_t_idx, ac_idx[col]]
                u_analog   = self._analog_to_u(raw_analog, col, sim)

                if self.jitter > 0:
                    u_analog = np.clip(
                        u_analog + self.rng.normal(0, self.jitter, size=u_analog.shape),
                        1e-6, 1 - 1e-6,
                    )

                out_arrays[col] = self._match_u(u_analog, su, sr)

            # Signal and forecast_signal directly from next_ens
            sig_col               = f"{site}{self.signal_suffix}"
            fcast_col             = f"{site}_forecast_signal"
            flat                  = next_ens[:, :, site_j].reshape(-1)
            out_arrays[sig_col]   = flat
            out_arrays[fcast_col] = flat

        idx = pd.MultiIndex.from_arrays([all_sim_ids, all_t_idx], names=["sim_id", "t"])
        return pd.DataFrame(out_arrays, index=idx).sort_index()

    # ------------------------------------------------------------------
    # Multi-event sample
    # ------------------------------------------------------------------

    def sample_events(
        self,
        resampled_all: pd.DataFrame,
        analog_payloads: np.ndarray,
        analog_joint_cols: List[str],
    ) -> pd.DataFrame:
        """
        Draw (duration, intensity) per event via univariate quantile matching.

        Parameters
        ----------
        resampled_all     : pd.DataFrame, MultiIndex (sim_id, t)
        analog_payloads   : np.ndarray, shape (E, T, n_joint_cols)
        analog_joint_cols : List[str]

        Returns
        -------
        pd.DataFrame, MultiIndex (sim_id, t, site, event_id)
        """
        if not self.multi_copula_sims:
            raise RuntimeError(
                "multi_copula_sims not provided at construction."
            )

        _, T_analog, _ = analog_payloads.shape
        ac_idx         = {col: i for i, col in enumerate(analog_joint_cols)}
        event_frames   = []

        for site in self.sites:
            if site not in self._sorted_u_multi:
                continue

            freq_col  = f"{site}_frequency"
            fcast_col = f"{site}_forecast_signal"

            if freq_col not in resampled_all.columns:
                continue

            freq     = resampled_all[freq_col].fillna(0).astype(int).clip(lower=0)
            freq_pos = freq[freq > 0]
            if freq_pos.empty:
                continue

            sim_m      = self.multi_copula_simulators[site]
            base_index = freq_pos.index.to_numpy()
            counts     = freq_pos.to_numpy()
            n_events   = int(counts.sum())
            rep_index  = np.repeat(base_index, counts)
            event_id   = np.concatenate([np.arange(1, k + 1, dtype=int) for k in counts])

            sim_ids_ev = np.array([ij[0] for ij in rep_index], dtype=int)
            t_vals_ev  = np.array([ij[1] for ij in rep_index], dtype=int)
            t_clamped  = np.minimum(t_vals_ev, T_analog - 1)

            samp_dict: Dict[str, np.ndarray] = {}

            for v in self.multi_vars:
                col = f"{site}{v}"
                if v == self.signal_suffix or col not in self._sorted_u_multi.get(site, {}):
                    continue

                su = self._sorted_u_multi[site][col]
                sr = self._sorted_raw_multi[site][col]

                if col not in ac_idx:
                    samp_dict[col] = np.full(n_events, np.median(sr))
                    continue

                raw_analog = analog_payloads[sim_ids_ev, t_clamped, ac_idx[col]]
                u_analog   = self._analog_to_u(raw_analog, col, sim_m)

                if self.jitter > 0:
                    u_analog = np.clip(
                        u_analog + self.rng.normal(0, self.jitter, size=u_analog.shape),
                        1e-6, 1 - 1e-6,
                    )

                samp_dict[col] = self._match_u(u_analog, su, sr)

            signal_per_month     = resampled_all.loc[freq_pos.index, fcast_col].to_numpy()
            samp_dict[fcast_col] = np.repeat(signal_per_month, counts)

            samp = pd.DataFrame(samp_dict)
            samp.index = pd.MultiIndex.from_arrays(
                [sim_ids_ev, t_vals_ev,
                 np.full(n_events, site, dtype=object), event_id],
                names=["sim_id", "t", "site", "event_id"],
            )
            event_frames.append(samp)

        if event_frames:
            return pd.concat(event_frames).sort_index()
        return pd.DataFrame(
            columns=["sim_id", "t", "site", "event_id"]
        ).set_index(["sim_id", "t", "site", "event_id"])
