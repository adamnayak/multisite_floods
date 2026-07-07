import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union
import matplotlib.pyplot as plt


def forecast_df_from_ens_member(ens_member: np.ndarray, sites: list[str], t_index=None) -> pd.DataFrame:
    df = pd.DataFrame(ens_member, columns=sites)
    if t_index is not None:
        df.index = t_index
        df.index.name = "t"
    return df


def z(x):
    mu = np.mean(x)
    sd = np.std(x)
    return (x - mu) / sd if sd > 0 else np.zeros_like(x)


def drop_constant_columns(df: pd.DataFrame, min_std: float = 1e-9) -> pd.DataFrame:
    std = df.std(axis=0, skipna=True)
    keep = std > min_std
    return df.loc[:, keep]


def tail_corr_matrix(df: pd.DataFrame, q: float = 0.9, min_pairs: int = 5) -> pd.DataFrame:
    """
    Pairwise 'tail correlation': Pearson r computed only on times
    when both series exceed their own q-quantile.

    df: columns are sites
    q:  quantile threshold (e.g., 0.9)
    """
    cols = df.columns
    tc = pd.DataFrame(index=cols, columns=cols, dtype=float)

    for i, ci in enumerate(cols):
        xi = df[ci]
        # Pre-compute tail threshold for site i
        ti = xi.quantile(q)

        for j, cj in enumerate(cols):
            if j < i:
                # Symmetric
                tc.iloc[i, j] = tc.iloc[j, i]
                continue

            xj = df[cj]
            tj = xj.quantile(q)

            # Common non-missing times
            mask = xi.notna() & xj.notna()
            xi_ = xi[mask]
            xj_ = xj[mask]

            if len(xi_) < min_pairs:
                tc.loc[ci, cj] = np.nan
                continue

            # Tail mask: both in upper tail
            tail_mask = (xi_ > ti) & (xj_ > tj)
            if tail_mask.sum() < min_pairs:
                tc.loc[ci, cj] = np.nan
                continue

            r = np.corrcoef(xi_[tail_mask], xj_[tail_mask])[0, 1]
            tc.loc[ci, cj] = r

    # Diagonal = 1 by convention (if enough tail points exist)
    for c in cols:
        if pd.isna(tc.loc[c, c]):
            tc.loc[c, c] = 1.0

    return tc

def plot_qq_multisite(variable_name, obs_dict, sim_dict, sites,
                      n_quantiles=200, quantile_clip=(0.001, 0.999)):
    """
    variable_name: str for title ("Signal", "Intensity", etc.)
    obs_dict: {site: array}
    sim_dict: {site: array}
    """

    n_sites = len(sites)
    fig, axes = plt.subplots(1, n_sites, figsize=(4*n_sites, 4), sharex=False, sharey=False)

    if n_sites == 1:
        axes = [axes]  # ensure iterable

    qs = np.linspace(quantile_clip[0], quantile_clip[1], n_quantiles)

    for ax, site in zip(axes, sites):
        x_obs = obs_dict[site]
        x_sim = sim_dict[site]

        if len(x_obs) < 10 or len(x_sim) < 10:
            ax.set_title(f"{site} (insufficient data)")
            ax.axis("off")
            continue

        # Normalize
        xo = (x_obs - x_obs.mean()) / x_obs.std()
        xs = (x_sim - x_sim.mean()) / x_sim.std()

        q_obs = np.quantile(xo, qs)
        q_sim = np.quantile(xs, qs)

        ax.scatter(q_obs, q_sim, s=12, alpha=0.7)
        lo = min(q_obs.min(), q_sim.min())
        hi = max(q_obs.max(), q_sim.max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)

        ax.set_title(site)
        ax.set_xlabel("Obs quantiles (z)")
        ax.set_ylabel("Sim quantiles (z)")

    fig.suptitle(f"{variable_name} – QQ Across Sites", fontsize=16)
    plt.tight_layout()
    plt.show()

def plot_qq_overlay(variable_name, obs_dict, sim_dict, sites,
                    n_quantiles=200, quantile_clip=(0.001,0.999)):
    
    qs = np.linspace(quantile_clip[0], quantile_clip[1], n_quantiles)

    plt.figure(figsize=(7, 6))

    for site in sites:
        x_obs = obs_dict[site]
        x_sim = sim_dict[site]

        if len(x_obs) < 10 or len(x_sim) < 10:
            continue

        xo = (x_obs - x_obs.mean()) / x_obs.std()
        xs = (x_sim - x_sim.mean()) / x_sim.std()

        q_obs = np.quantile(xo, qs)
        q_sim = np.quantile(xs, qs)

        # Light line per site
        plt.plot(q_obs, q_sim, alpha=0.25, lw=1)

    # Reference line
    lo = min(q_obs.min(), q_sim.min())
    hi = max(q_obs.max(), q_sim.max())
    plt.plot([lo, hi], [lo, hi], "k--", lw=1)

    plt.title(f"{variable_name} – QQ (All Sites Overlaid)")
    plt.xlabel("Obs Z-Quantile")
    plt.ylabel("Sim Z-Quantile")
    plt.show()

@dataclass
class KNNKernel:
    """
    Kernel weights for KNN resampling.

    Rank-based kernels:
      - uniform_k:      w_i = 1 / k
      - inverse_rank:   w_i ∝ 1 / rank_i
      - exp_rank:       w_i ∝ exp(-alpha * (rank_i - 1))

    """
    kind: str = "inverse_rank"
    alpha: float = 1.0
    sigma: float = 1.0

    def weights(self, k: int, distances: np.ndarray | None = None) -> np.ndarray:
        if k <= 0:
            raise ValueError("k must be >= 1")

        # ------------------
        # Rank-based kernels
        # ------------------
        if self.kind == "uniform_k":
            w = np.ones(k)

        elif self.kind == "inverse_rank":
            r = np.arange(1, k + 1, dtype=float)
            w = 1.0 / r

        elif self.kind == "exp_rank":
            r = np.arange(1, k + 1, dtype=float)
            w = np.exp(-self.alpha * (r - 1.0))
        else:
            raise ValueError(f"Unknown kernel kind: {self.kind}")

        return w / w.sum()


class KNNConditionalCopulaSampler:
    """
    Conditional sampling from multiple empirical copula simulation tables using KNN on a 'signal' column.

    Inputs
    ------
    copula_sims: dict[str, pd.DataFrame]
        Keys are site names; values are simulated copula tables (rows = simulated months/events).
        Each table must contain the signal column (last variable) used for conditioning.

    signal_col: str
        Column name within each copula table used as conditioning variable.
        (e.g., f"{site}_max_z_wave" or standardized "max_z_wave")

    kernel: KNNKernel
        Defines rank-based sampling probabilities among the k nearest neighbors.

    Behavior
    --------
    For each forecasted signal value, we:
      1) find k nearest neighbors in that site's simulated signal column
      2) sample one neighbor index with kernel weights (rank-based)
      3) take the full row (all variables) from the copula table
      4) overwrite / attach the forecast signal value (optional)

    Output
    ------
    DataFrame with one sampled row per forecast step per site (stacked).
    """

    def __init__(
        self,
        copula_sims: Dict[str, pd.DataFrame],
        signal_col_by_site: Dict[str, str],
        k: Optional[Union[int, Dict[str, int]]] = None,
        kernel: Optional[KNNKernel] = None,
        attach_forecast_signal: bool = True,
        forecast_signal_colname: str = "forecast_signal",
        random_state: Optional[Union[int, np.random.Generator]] = None,
    ):
        self.copula_sims = copula_sims
        self.k = k
        self.signal_col_by_site = signal_col_by_site
        self.kernel = kernel or KNNKernel(kind="inverse_rank")
        self.attach_forecast_signal = attach_forecast_signal
        self.forecast_signal_colname = forecast_signal_colname
        self.rng = (
            np.random.default_rng(random_state)
            if not isinstance(random_state, np.random.Generator)
            else random_state
        )

        # precompute arrays for fast KNN lookup
        self._sig_values = {}
        self._df_values = {}
        self._df_cols = {}
        for site, df in self.copula_sims.items():
            sig_col = self.signal_col_by_site[site]
            if sig_col not in df.columns:
                raise KeyError(f"[{site}] signal column '{sig_col}' not found in copula_sims dataframe.")

            self._sig_values[site] = df[sig_col].to_numpy(dtype=float)
            self._df_values[site] = df.to_numpy()
            self._df_cols[site] = list(df.columns)

    def _knn_indices_1d(self, x: float, pool: np.ndarray, k: int) -> np.ndarray:
        """
        Return indices of k nearest neighbors in 1D pool to value x (by absolute distance).
        Uses argpartition for efficiency.
        """
        d = np.abs(pool - x)
        if k >= d.size:
            # all points are neighbors
            if k > d.size:
                print("FYI: k larger than d, returning all")
            return np.argsort(d)
        idx = np.argpartition(d, kth=k - 1)[:k]
        # sort these k by distance
        idx = idx[np.argsort(d[idx])]
        return idx
    
    def sample(
        self,
        next_forecast,   # np.ndarray or pd.DataFrame as we discussed
        sites=None,
        return_format: str = "wide",   # "wide" or "long"
    ) -> pd.DataFrame:
        """
        next_forecast:
          - np.ndarray shape (T, S): columns correspond to `sites`
          - pd.DataFrame: columns are site names; index is time (preserved)
        """
        # --- If DataFrame, extract sites + array and remember time index ---
        time_index = None
        if isinstance(next_forecast, pd.DataFrame):
            next_forecast_df = next_forecast
            sites = list(next_forecast_df.columns) if sites is None else list(sites)
            time_index = next_forecast_df.index
            X = next_forecast_df[sites].to_numpy(dtype=float)
        else:
            X = np.asarray(next_forecast, dtype=float)
    
        # --- Core logic expects ndarray + sites ---
        if sites is None:
            sites = list(self.copula_sims.keys())
    
        if X.ndim != 2:
            raise ValueError("next_forecast must be 2D array or DataFrame.")
        T, S = X.shape
        if S != len(sites):
            raise ValueError(f"next_forecast has {S} columns but sites has {len(sites)} entries.")
    
        # Collect per-timestep dicts (wide) OR per-(t,site) rows (long)
        if return_format not in {"wide", "long"}:
            raise ValueError("return_format must be 'wide' or 'long'")
    
        wide_rows = [dict() for _ in range(T)]
        long_rows = []
        long_index = []
    
        for j, site in enumerate(sites):
            df_vals = self._df_values[site]
            sig_pool = self._sig_values[site]
            cols = self._df_cols[site]
    
            k_site = self.k[site] if isinstance(self.k, dict) else self.k
            k_eff = min(int(k_site), sig_pool.size)
            w = self.kernel.weights(k_eff)
    
            for t in range(T):
                x = X[t,j] #sig_pool[t] for historic, X[t,j] for prediction
                nn = self._knn_indices_1d(x, sig_pool, k=k_eff)
                pick = self.rng.choice(nn, p=w) # for fully random pick = self.rng.integers(0, sig_pool.size)
                row = df_vals[pick].copy()
    
                row_dict = dict(zip(cols, row))
                # attach forecast signal per site
                if self.attach_forecast_signal:
                    row_dict[f"{site}_{self.forecast_signal_colname}"] = x
                    
                if return_format == "wide":
                    # Merge all site variables into the timestep row dict.
                    # Columns already site-prefixed in your copula_sims, so no collisions.
                    wide_rows[t].update(row_dict)
                else:
                    long_rows.append(row_dict)
                    long_index.append((t, site))
    
        if return_format == "wide":
            out_df = pd.DataFrame(wide_rows)
    
            # preserve time index if you passed a forecast DataFrame
            if isinstance(next_forecast, pd.DataFrame):
                out_df.index = next_forecast.index
                out_df.index.name = "t"
            else:
                out_df.index = pd.RangeIndex(T, name="t")
    
            return out_df
    
        # long format
        out_df = pd.DataFrame(long_rows)
        out_df.index = pd.MultiIndex.from_tuples(long_index, names=["t", "site"])
        return out_df.sort_index()


    def plot_distributions_compare(
        self,
        observed: pd.DataFrame,
        simulated: pd.DataFrame,
        cols: Optional[Sequence[str]] = None,
        *,
        sim_id_level: str = "sim_id",
        max_sim_rows: Optional[int] = 200_000,
        bins: int = 60,
        quantile_clip: Tuple[float, float] = (0.001, 0.999),
        density: bool = True,
        figsize: Tuple[int, int] = (14, 10),
        suptitle: Optional[str] = None,
        sharex: bool = False,
    ):
        """
        Compare marginal distributions between observed data and simulated samples.

        Parameters
        ----------
        observed : pd.DataFrame
            Observed dataframe (e.g., climate_filtered[VAR_COLS] or a wide table with site-prefixed cols).

        simulated : pd.DataFrame
            Simulated dataframe (e.g., resampled_all from concat with keys=["sim_id","t"] or a single sim output).
            If it has a MultiIndex with sim_id_level, we flatten it.

        cols : list[str], optional
            Columns to plot. If None, uses intersection of observed and simulated columns.

        sim_id_level : str
            Name of the MultiIndex level for simulation id in `simulated` (default "sim_id").

        max_sim_rows : int, optional
            Downsample simulated rows for faster plotting (recommended if you have many sims).

        bins : int
            Histogram bins.

        quantile_clip : (low, high)
            Clip both distributions to these quantiles (computed on observed+sim combined) to make plots readable.

        density : bool
            If True, plot normalized histograms.

        sharex : bool
            If True, share x-axis across panels.

        Notes
        -----
        - This plots histograms (not KDE) to keep dependencies light and fast.
        - It’s intended for marginal checks, not joint dependence diagnostics.
        """
        # ---- sanitize simulated index (common: MultiIndex [sim_id, t]) ----
        sim_df = simulated.copy()
        if isinstance(sim_df.index, pd.MultiIndex) and sim_id_level in sim_df.index.names:
            # drop sim_id from index; keep t if present
            sim_df = sim_df.reset_index(level=sim_id_level, drop=True)

        obs_df = observed.copy()

        # ---- choose columns ----
        if cols is None:
            cols = sorted(set(obs_df.columns).intersection(set(sim_df.columns)))
        cols = list(cols)
        if len(cols) == 0:
            raise ValueError("No overlapping columns between observed and simulated to plot.")

        # ---- optional downsample simulated ----
        if max_sim_rows is not None and len(sim_df) > max_sim_rows:
            sim_df = sim_df.sample(n=max_sim_rows, replace=False, random_state=0)

        # ---- layout ----
        n = len(cols)
        ncols = 2 if n > 1 else 1
        nrows = int(np.ceil(n / ncols))

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, sharex=sharex)
        axes = np.atleast_1d(axes).ravel()

        for i, c in enumerate(cols):
            ax = axes[i]
            x_obs = pd.to_numeric(obs_df[c], errors="coerce").to_numpy()
            x_sim = pd.to_numeric(sim_df[c], errors="coerce").to_numpy()

            x_obs = x_obs[np.isfinite(x_obs)]
            x_sim = x_sim[np.isfinite(x_sim)]

            if x_obs.size == 0 or x_sim.size == 0:
                ax.set_title(f"{c} (empty)")
                continue

            # clip to comparable range for readability
            x_all = np.concatenate([x_obs, x_sim])
            lo = np.quantile(x_all, quantile_clip[0])
            hi = np.quantile(x_all, quantile_clip[1])
            if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                x_obs_plot = x_obs[(x_obs >= lo) & (x_obs <= hi)]
                x_sim_plot = x_sim[(x_sim >= lo) & (x_sim <= hi)]
            else:
                x_obs_plot, x_sim_plot = x_obs, x_sim

            ax.hist(x_obs_plot, bins=bins, density=density, alpha=0.5, label="Observed")
            ax.hist(x_sim_plot, bins=bins, density=density, alpha=0.5, label="Simulated")
            ax.set_title(c)
            ax.grid(True, alpha=0.2)

            # quick numeric summary in the corner
            obs_p = np.quantile(x_obs, [0.5, 0.9, 0.99])
            sim_p = np.quantile(x_sim, [0.5, 0.9, 0.99])
            txt = (
                f"med: {obs_p[0]:.3g} | {sim_p[0]:.3g}\n"
                f"p90: {obs_p[1]:.3g} | {sim_p[1]:.3g}\n"
                f"p99: {obs_p[2]:.3g} | {sim_p[2]:.3g}"
            )
            ax.text(0.98, 0.98, txt, transform=ax.transAxes, va="top", ha="right", fontsize=9)

            if i == 0:
                ax.legend()

        # hide unused axes
        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        if suptitle:
            fig.suptitle(suptitle, y=0.99)

        fig.tight_layout()
        return fig

    def filter_events_by_intensity(self, df: pd.DataFrame, thr_map: dict) -> pd.DataFrame:
        """
        Keep rows where at least one site's max_intensity exceeds its threshold.
    
        Parameters
        ----------
        df : pd.DataFrame
            Wide dataframe with columns like "{site}_max_intensity".
    
        thr_map : dict
            Mapping {site: threshold}.
    
        Returns
        -------
        pd.DataFrame
            Filtered dataframe.
        """
        # Build boolean mask per site
        masks = []
        for site, thr in thr_map.items():
            col = f"{site}_max_intensity"
            if col not in df.columns:
                raise KeyError(f"Column '{col}' not found in dataframe.")
            masks.append(df[col] >= thr)
            
        # Inclusive OR across sites
        keep_mask = np.logical_or.reduce(masks)
        
        return df.loc[keep_mask].copy()
