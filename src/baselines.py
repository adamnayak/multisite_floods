"""
baselines.py
============
Three stochastic flood generation baselines for comparison against the
transformer joint analog framework. All baselines:
  - Are fit on training data only
  - Generate T-month synthetic sequences without accessing test observations
  - Share a common evaluate() interface returning (crps_df, qq_data)
  - Use identical event extraction (extract_events_from_daily) and CRPS
    helpers for fair comparison

Baselines
---------
1. LallSharmaKNN   — Lall & Sharma (1996) kNN Markov successor generator
2. SeasonalAR      — Periodic AR(2) on monthly max anomalies
                     (Salas et al. 1980; Stedinger & Taylor 1982 WRR)
3. NeymanScott     — Independent parametric sampling per month
                     Poisson(frequency), Exponential(intensity),
                     Exponential(duration) fit to training record

Shared helpers
--------------
load_split, compute_monthly_max, compute_thresholds,
extract_events_from_daily, extract_events_monthly,
build_daily_lookup, _crps_energy, _crps_energy_fast,
_spread_subsample, summarise_crps, compute_pooled_sequence_crps
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from scipy import stats


# ══════════════════════════════════════════════════════════════════════════════
# Shared data helpers
# ══════════════════════════════════════════════════════════════════════════════

def load_split(
    train_path: str | Path,
    test_path:  str | Path,
    date_col:   str = "datetime",
) -> Tuple[pd.DataFrame, pd.DataFrame, list]:
    """Load train/test daily CSVs. Sites inferred from column headers."""
    train = pd.read_csv(train_path, parse_dates=[date_col], index_col=date_col)
    test  = pd.read_csv(test_path,  parse_dates=[date_col], index_col=date_col)
    sites = [c for c in train.columns if c != date_col]
    return train, test[sites], sites


def compute_monthly_max(daily: pd.DataFrame) -> pd.DataFrame:
    """Resample daily to monthly max with PeriodIndex(freq='M')."""
    mm = daily.resample("M").max()
    mm.index = mm.index.to_period("M")
    return mm


def compute_thresholds(monthly_max_train: pd.DataFrame) -> Dict[str, float]:
    """Per-site threshold = mean monthly max over training period."""
    return monthly_max_train.mean(axis=0).to_dict()


def extract_events_from_daily(
    daily_series: np.ndarray,
    threshold:    float,
) -> Dict[str, float]:
    """
    Extract flood event statistics from a daily flow array.
      Frequency : count of contiguous threshold exceedance runs
      Intensity : max peak across runs (matches model max_intensity)
      Duration  : mean run length in days
    """
    above = daily_series > threshold
    if not above.any():
        return {"Frequency": 0.0, "Intensity": 0.0, "Duration": 0.0}

    runs, in_run, start = [], False, 0
    for i, flag in enumerate(above):
        if flag and not in_run:
            in_run, start = True, i
        elif not flag and in_run:
            in_run = False
            runs.append((start, i - 1))
    if in_run:
        runs.append((start, len(above) - 1))

    if not runs:
        return {"Frequency": 0.0, "Intensity": 0.0, "Duration": 0.0}

    peaks     = [daily_series[s:e+1].max() for s, e in runs]
    durations = [e - s + 1               for s, e in runs]
    return {
        "Frequency": float(len(runs)),
        "Intensity": float(np.max(peaks)),
        "Duration":  float(np.mean(durations)),
    }


def extract_events_monthly(
    daily:   pd.DataFrame,
    thr_map: Dict[str, float],
) -> pd.DataFrame:
    """Extract observed flood stats for every (site, month). Period[M] index."""
    records = []
    for site, thr in thr_map.items():
        if site not in daily.columns:
            continue
        series = daily[site].dropna()
        for period, grp in series.groupby(series.index.to_period("M")):
            ev = extract_events_from_daily(grp.values, thr)
            records.append({
                "Month": pd.Period(period, freq="M"),
                "Site":  site, **ev,
            })
    return pd.DataFrame(records)


def build_daily_lookup(
    train_daily: pd.DataFrame,
    sites:       List[str],
) -> Dict[Tuple[str, object], np.ndarray]:
    """Pre-build (site, Period[M]) -> daily array lookup for fast retrieval."""
    lookup = {}
    for site in sites:
        if site not in train_daily.columns:
            continue
        series = train_daily[site]
        for period, grp in series.groupby(series.index.to_period("M")):
            lookup[(site, pd.Period(period, freq="M"))] = grp.values
    return lookup


# ══════════════════════════════════════════════════════════════════════════════
# Shared CRPS helpers
# ══════════════════════════════════════════════════════════════════════════════

def _crps_energy(members: np.ndarray, obs: float) -> float:
    """Full energy-form CRPS. Use only for small member arrays."""
    if np.isnan(obs) or len(members) == 0:
        return np.nan
    M     = len(members)
    term1 = np.mean(np.abs(members - obs))
    term2 = np.sum(np.abs(members[:, None] - members[None, :])) / (2.0 * M * M)
    return float(term1 - term2)


def _spread_subsample(members: np.ndarray, n_sub: int = 2000) -> float:
    """Estimate E|X-X'|/2 via random subsampling. Avoids O(M^2) matrix."""
    rng  = np.random.default_rng(0)
    idx1 = rng.integers(0, len(members), size=n_sub)
    idx2 = rng.integers(0, len(members), size=n_sub)
    return float(np.mean(np.abs(members[idx1] - members[idx2])) / 2.0)


def _crps_energy_fast(members: np.ndarray, obs_val: float, spread: float) -> float:
    """CRPS with precomputed spread — O(M) not O(M^2)."""
    return float(np.mean(np.abs(members - obs_val)) - spread)


def summarise_crps(crps_df: pd.DataFrame, label: str = "") -> pd.DataFrame:
    """Print and return mean CRPS per variable."""
    summary = (
        crps_df.groupby("variable")["CRPS"]
        .agg(mean="mean", median="median", n="count")
        .reset_index()
    )
    title = f"CRPS — {label}" if label else "CRPS"
    print(f"\n=== {title} ===")
    print(summary.to_string(index=False))
    return summary


def compute_pooled_sequence_crps(
    ensemble:     Dict[str, np.ndarray],
    events_test:  pd.DataFrame,
    periods_test: pd.PeriodIndex,
    variables:    List[str] = ["Frequency", "Intensity", "Duration"],
    n_sub:        int       = 2000,
    label:        str       = "",
) -> pd.DataFrame:
    """
    Pooled sequence CRPS — insurance-relevant metric.

    For each (site, variable):
      - Pool all T * n_sims simulated values into one distribution
      - Score each T observed months against this distribution
      - Average CRPS across T observations

    Captures whether the full simulated sequence distribution covers
    the range of observed outcomes regardless of exact timing.
    Memory safe: spread precomputed via subsampling.

    Parameters
    ----------
    ensemble : {site: np.ndarray (T, n_sims, 3)}
               axis 2 = [Frequency, Intensity, Duration]
    """
    var_idx = {v: i for i, v in
               enumerate(["Frequency", "Intensity", "Duration"])}

    et = events_test.copy()
    et["Month"] = pd.PeriodIndex(
        [pd.Period(p, freq="M") for p in et["Month"]], freq="M"
    )
    obs_idx = et.set_index(["Site", "Month"])
    records = []

    for site, ens in ensemble.items():
        if site not in obs_idx.index.get_level_values("Site"):
            continue
        site_obs = obs_idx.xs(site, level="Site")

        for v in variables:
            vi = var_idx.get(v)
            if vi is None:
                continue

            pooled_sim = ens[:, :, vi].ravel()
            if len(pooled_sim) < 10:
                continue

            spread = _spread_subsample(pooled_sim, n_sub=n_sub)

            crps_vals = []
            for period in periods_test:
                if period not in site_obs.index:
                    continue
                obs_row = site_obs.loc[period]
                if isinstance(obs_row, pd.DataFrame):
                    obs_row = obs_row.iloc[0]
                obs_val = float(obs_row[v]) if v in obs_row.index else np.nan
                if np.isnan(obs_val):
                    continue
                crps_vals.append(_crps_energy_fast(pooled_sim, obs_val, spread))

            if not crps_vals:
                continue

            records.append({
                "Site":     site,
                "variable": v,
                "CRPS":     float(np.mean(crps_vals)),
                "n_obs":    len(crps_vals),
                "ens_mean": float(pooled_sim.mean()),
                "ens_p10":  float(np.percentile(pooled_sim, 10)),
                "ens_p90":  float(np.percentile(pooled_sim, 90)),
            })

    df = pd.DataFrame(records)
    title = f"Pooled Sequence CRPS — {label}" if label else "Pooled Sequence CRPS"
    print(f"\n=== {title} ===")
    if not df.empty:
        print(
            df.groupby("variable")["CRPS"]
            .agg(mean="mean", median="median", n="count")
            .reset_index()
            .to_string(index=False)
        )
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Shared evaluate() mixin — inherited by all three baselines
# ══════════════════════════════════════════════════════════════════════════════

class _BaselineEvaluator:
    """
    Shared evaluate() method for all baselines.
    Subclasses must implement generate(T, n_sims) returning
    {site: np.ndarray (T, n_sims, 3)} where axis 2 = [Freq, Int, Dur].
    """

    def evaluate(
        self,
        ensemble:      Dict[str, np.ndarray],
        events_test:   pd.DataFrame,
        periods_test:  pd.PeriodIndex,
        variables:     List[str]            = ["Frequency", "Intensity", "Duration"],
        n_q:           int                  = 100,
        lo_q:          float                = 0.10,
        hi_q:          float                = 0.90,
        events_qq_ref: Optional[pd.DataFrame] = None,
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Compute monthly CRPS and QQ envelope from a pre-generated ensemble.

        CRPS   : per (Site, Month, variable) — test period observed reference
        QQ     : per-site z-scored envelope  — training record obs reference
                 (if events_qq_ref provided, else falls back to events_test)
        """
        var_idx  = {v: i for i, v in
                    enumerate(["Frequency", "Intensity", "Duration"])}
        qs       = np.linspace(0.01, 0.99, n_q)
        pos_only = {"Frequency": False, "Intensity": True, "Duration": True}

        # normalise observed CRPS reference
        et = events_test.copy()
        et["Month"] = pd.PeriodIndex(
            [pd.Period(p, freq="M") for p in et["Month"]], freq="M"
        )
        obs_idx = et.set_index(["Site", "Month"])[variables]

        # QQ reference — training record if provided
        qq_ref = events_qq_ref if events_qq_ref is not None else events_test
        et_qq  = qq_ref.copy()
        et_qq["Month"] = pd.PeriodIndex(
            [pd.Period(p, freq="M") for p in et_qq["Month"]], freq="M"
        )
        qq_obs_idx = et_qq.set_index(["Site", "Month"])[variables]

        crps_records = []
        q_obs_by_var = {v: [] for v in variables}
        q_sim_by_var = {v: [] for v in variables}

        for site, ens in ensemble.items():
            # ── monthly CRPS ──────────────────────────────────────────
            if site in obs_idx.index.get_level_values("Site"):
                site_obs = obs_idx.xs(site, level="Site")
                for t_idx, period in enumerate(periods_test):
                    if period not in site_obs.index:
                        continue
                    obs_row = site_obs.loc[period]
                    if isinstance(obs_row, pd.DataFrame):
                        obs_row = obs_row.iloc[0]
                    for v in variables:
                        vi = var_idx.get(v)
                        if vi is None:
                            continue
                        obs_val = float(obs_row[v]) \
                                  if v in obs_row.index else np.nan
                        if np.isnan(obs_val):
                            continue
                        members = ens[t_idx, :, vi]
                        crps_records.append({
                            "Site":     site,
                            "Month":    period,
                            "variable": v,
                            "CRPS":     _crps_energy(members, obs_val),
                        })

            # ── QQ vectors ────────────────────────────────────────────
            for v in variables:
                vi = var_idx.get(v)
                if vi is None:
                    continue
                po = pos_only.get(v, False)

                if site in qq_obs_idx.index.get_level_values("Site"):
                    h_obs = qq_obs_idx.xs(site, level="Site")[v].dropna().values
                else:
                    h_obs = np.array([])
                if po:
                    h_obs = h_obs[h_obs > 0]

                h_sim = ens[:, :, vi].ravel()
                if po:
                    h_sim = h_sim[h_sim > 0]

                if len(h_obs) < 5 or len(h_sim) < 5:
                    continue
                mu, sigma = h_obs.mean(), h_obs.std()
                if sigma < 1e-9:
                    continue

                q_obs_by_var[v].append(np.quantile((h_obs - mu) / sigma, qs))
                q_sim_by_var[v].append(np.quantile((h_sim - mu) / sigma, qs))

        # aggregate QQ
        qq_data = {}
        for v in variables:
            if not q_obs_by_var[v]:
                qq_data[v] = None
                continue
            q_obs = np.array(q_obs_by_var[v])
            q_sim = np.array(q_sim_by_var[v])
            qq_data[v] = {
                "qs":         qs,
                "obs_median": np.median(q_obs, axis=0),
                "sim_median": np.median(q_sim, axis=0),
                "sim_lo":     np.quantile(q_sim, lo_q, axis=0),
                "sim_hi":     np.quantile(q_sim, hi_q, axis=0),
            }

        print(f"CRPS records: {len(crps_records):,}")
        return pd.DataFrame(crps_records), qq_data


# ══════════════════════════════════════════════════════════════════════════════
# Baseline 1 — Lall & Sharma (1996) kNN Markov successor
# ══════════════════════════════════════════════════════════════════════════════

class LallSharmaKNN(_BaselineEvaluator):
    """
    Lall & Sharma (1996) stochastic streamflow generator.

    Self-propagating Markov successor: at each step, K nearest training
    months are found by standardised monthly max distance, one is sampled
    with inverse-rank weights, and its SUCCESSOR in the training record
    becomes the next simulated value. Test data never accessed.

    Reference: Lall, U. & Sharma, A. (1996). A Nearest Neighbor Bootstrap
    for Resampling Hydrologic Time Series. Water Resources Research, 32(3).
    """

    def __init__(
        self,
        K:          Optional[int] = None,
        lag_months: int           = 1,
        seed:       Optional[int] = 42,
    ):
        self.K_override = K
        self.lag_months = lag_months
        self.rng        = np.random.default_rng(seed)
        self._fitted    = False

    def fit(
        self,
        train_daily:       pd.DataFrame,
        monthly_max_train: pd.DataFrame,
        thr_map:           Dict[str, float],
    ) -> "LallSharmaKNN":
        self.thr_map = thr_map
        self.sites   = list(monthly_max_train.columns)

        N      = len(monthly_max_train)
        self.K = self.K_override if self.K_override is not None \
                 else int(np.floor(np.sqrt(N)))

        self.mu_    = monthly_max_train.mean()
        self.sigma_ = monthly_max_train.std().replace(0, 1.0)
        self.periods_train = pd.PeriodIndex(monthly_max_train.index, freq="M")

        mm_norm           = ((monthly_max_train - self.mu_) / self.sigma_).values
        self.X_train      = self._build_feature_matrix(mm_norm, self.lag_months)
        self.mm_norm_train = mm_norm

        ranks        = np.arange(1, self.K + 1, dtype=float)
        self.weights = (1.0 / ranks) / (1.0 / ranks).sum()

        print("Building daily sequence lookup ...")
        self.daily_lookup = build_daily_lookup(train_daily, self.sites)
        print(f"  {len(self.daily_lookup):,} entries cached.")

        self._fitted = True
        return self

    @staticmethod
    def _build_feature_matrix(mm_norm: np.ndarray, lag: int) -> np.ndarray:
        N, S = mm_norm.shape
        X    = np.zeros((N, S * lag), dtype=float)
        for l in range(lag):
            sc, ec = l * S, l * S + S
            X[l:, sc:ec] = mm_norm[:N-l] if l > 0 else mm_norm
        return X

    def generate(
        self,
        T:      int,
        n_sims: int = 1000,
    ) -> Dict[str, np.ndarray]:
        """Generate n_sims sequences of length T via Markov successor."""
        assert self._fitted
        N       = len(self.periods_train)
        results = {}
        n_sites = len(self.sites)

        for s_idx, site in enumerate(self.sites):
            print(f"  [{s_idx+1:>3}/{n_sites}] {site[:60]}", end="\r")
            thr           = self.thr_map.get(site)
            site_feat_idx = [l * len(self.sites) + s_idx
                             for l in range(self.lag_months)]
            ens = np.zeros((T, n_sims, 3))

            for sim_i in range(n_sims):
                current_norm = self.mm_norm_train[-self.lag_months:, s_idx].copy()

                for t in range(T):
                    x_q   = current_norm[::-1]
                    dists = np.linalg.norm(
                        self.X_train[:, site_feat_idx] - x_q, axis=1
                    )
                    dists[N - 1] = np.inf   # no successor for last month
                    knn_idx  = np.argsort(dists)[:self.K]
                    chosen   = int(self.rng.choice(knn_idx, p=self.weights))
                    succ_period = self.periods_train[chosen + 1]
                    daily_seq   = self.daily_lookup.get((site, succ_period))

                    if daily_seq is None or len(daily_seq) == 0:
                        ev, succ_norm = {"Frequency": 0.0, "Intensity": 0.0,
                                         "Duration": 0.0}, 0.0
                    else:
                        ev        = extract_events_from_daily(daily_seq, thr)
                        raw_val   = daily_seq.max()
                        succ_norm = (raw_val - float(self.mu_[site])) \
                                    / float(self.sigma_[site])

                    ens[t, sim_i, 0] = ev["Frequency"]
                    ens[t, sim_i, 1] = ev["Intensity"]
                    ens[t, sim_i, 2] = ev["Duration"]
                    current_norm = np.append(current_norm[1:], succ_norm)

            results[site] = ens

        print(f"\nKNN done. {n_sims}×{T}×{n_sites} ensemble generated.")
        return results


# ══════════════════════════════════════════════════════════════════════════════
# Baseline 2 — Periodic AR(2) on monthly max anomalies
# ══════════════════════════════════════════════════════════════════════════════

class SeasonalAR(_BaselineEvaluator):
    """
    Periodic AR(2) stochastic streamflow generator.

    Fits an AR(2) model on calendar-month anomalies (deviations from
    monthly mean) of monthly maximum streamflow using OLS. Generates
    synthetic sequences by self-propagating from the last two training
    months, adding Gaussian noise with the residual standard deviation.
    Flood characteristics are extracted from the daily sequences of
    the nearest training month to the simulated monthly max value.

    This is the standard PAR(2) model for monthly streamflow generation.
    References:
      Salas, J.D. et al. (1980). Applied Modeling of Hydrologic Time Series.
        Water Resources Publications, Littleton, CO.
      Stedinger, J.R. & Taylor, M.R. (1982). Synthetic streamflow generation.
        Water Resources Research, 18(4), 919-924.

    The model fits AR(2) on anomalies:
        a_t = x_t - mu(m_t)
        a_t = phi1 * a_{t-1} + phi2 * a_{t-2} + eps_t
        eps_t ~ N(0, sigma_eps^2)
    where mu(m) is the calendar-month mean.
    """

    def __init__(self, seed: Optional[int] = 42):
        self.rng     = np.random.default_rng(seed)
        self._fitted = False

    def fit(
        self,
        train_daily:       pd.DataFrame,
        monthly_max_train: pd.DataFrame,
        thr_map:           Dict[str, float],
    ) -> "SeasonalAR":
        self.thr_map = thr_map
        self.sites   = list(monthly_max_train.columns)
        self.periods_train = pd.PeriodIndex(monthly_max_train.index, freq="M")

        # store last two training values per site for initialisation
        self.last_two = {}
        self.params   = {}   # {site: {month_means, phi1, phi2, sigma_eps}}

        print("Fitting SAR(2) per site ...")
        for site in self.sites:
            if site not in monthly_max_train.columns:
                continue

            x = monthly_max_train[site].values.astype(float)   # (N,)
            months = np.array([p.month for p in self.periods_train])
            N      = len(x)

            # calendar-month means from training data
            month_means = {}
            for m in range(1, 13):
                vals = x[months == m]
                month_means[m] = float(np.nanmean(vals)) if len(vals) > 0 else 0.0

            # compute anomalies
            mu_arr = np.array([month_means[m] for m in months])
            a      = x - mu_arr   # anomaly series

            # OLS for AR(2): a_t = phi1*a_{t-1} + phi2*a_{t-2} + eps
            # build design matrix from lag-2 onwards
            if N < 4:
                phi1, phi2, sigma_eps = 0.0, 0.0, float(np.nanstd(a))
            else:
                Y = a[2:]               # targets
                A = np.column_stack([a[1:-1], a[:-2]])  # [a_{t-1}, a_{t-2}]
                mask = np.isfinite(Y) & np.all(np.isfinite(A), axis=1)
                if mask.sum() < 3:
                    phi1, phi2, sigma_eps = 0.0, 0.0, float(np.nanstd(a))
                else:
                    coeffs, *_ = np.linalg.lstsq(A[mask], Y[mask], rcond=None)
                    phi1, phi2 = float(coeffs[0]), float(coeffs[1])
                    resid      = Y[mask] - A[mask] @ coeffs
                    sigma_eps  = float(np.std(resid))

            self.params[site] = {
                "month_means": month_means,
                "phi1":        phi1,
                "phi2":        phi2,
                "sigma_eps":   max(sigma_eps, 1e-6),
            }

            # store last two anomalies for warm-starting generation
            last_months   = [p.month for p in self.periods_train[-2:]]
            last_mu       = [month_means[m] for m in last_months]
            self.last_two[site] = (
                float(x[-2] - last_mu[0]) if N >= 2 else 0.0,
                float(x[-1] - last_mu[1]) if N >= 1 else 0.0,
            )

        # build daily lookup for event extraction
        print("Building daily sequence lookup ...")
        self.daily_lookup  = build_daily_lookup(train_daily, self.sites)
        self.mm_train      = monthly_max_train.copy()
        print(f"  {len(self.daily_lookup):,} entries cached.")

        self._fitted = True
        return self

    def _nearest_training_month(self, site: str, x_sim: float) -> object:
        """Find training month with monthly max closest to x_sim."""
        if site not in self.mm_train.columns:
            return None
        vals  = self.mm_train[site].values
        dists = np.abs(vals - x_sim)
        idx   = int(np.argmin(dists))
        return self.periods_train[idx]

    def generate(
        self,
        T:      int,
        n_sims: int = 1000,
    ) -> Dict[str, np.ndarray]:
        """
        Generate n_sims sequences of length T months per site.
        Self-propagates from last two training months using AR(2) recursion.
        Event extraction via nearest training month lookup.
        """
        assert self._fitted
        results = {}
        n_sites = len(self.sites)

        for s_idx, site in enumerate(self.sites):
            print(f"  [{s_idx+1:>3}/{n_sites}] {site[:60]}", end="\r")
            p   = self.params.get(site)
            thr = self.thr_map.get(site)
            if p is None or thr is None:
                continue

            phi1, phi2    = p["phi1"], p["phi2"]
            sigma_eps     = p["sigma_eps"]
            month_means   = p["month_means"]
            a_prev2, a_prev1 = self.last_two[site]

            # forecast month sequence (calendar months, wrapping)
            last_train_month = self.periods_train[-1].month
            forecast_months  = [
                ((last_train_month - 1 + t) % 12) + 1
                for t in range(1, T + 1)
            ]

            ens = np.zeros((T, n_sims, 3))

            for sim_i in range(n_sims):
                a1, a2 = a_prev2, a_prev1   # warm start from training

                for t in range(T):
                    m       = forecast_months[t]
                    mu_m    = month_means.get(m, 0.0)
                    eps     = self.rng.normal(0.0, sigma_eps)
                    a_new   = phi1 * a2 + phi2 * a1 + eps
                    x_sim   = mu_m + a_new
                    x_sim   = max(x_sim, 0.0)   # streamflow non-negative

                    # retrieve nearest training month's daily sequence
                    period  = self._nearest_training_month(site, x_sim)
                    daily_seq = self.daily_lookup.get((site, period)) \
                                if period is not None else None

                    if daily_seq is None or len(daily_seq) == 0:
                        ev = {"Frequency": 0.0, "Intensity": 0.0, "Duration": 0.0}
                    else:
                        ev = extract_events_from_daily(daily_seq, thr)

                    ens[t, sim_i, 0] = ev["Frequency"]
                    ens[t, sim_i, 1] = ev["Intensity"]
                    ens[t, sim_i, 2] = ev["Duration"]
                    a1, a2 = a2, a_new   # shift lag window

            results[site] = ens

        print(f"\nSAR done. {n_sims}×{T}×{n_sites} ensemble generated.")
        return results


# ══════════════════════════════════════════════════════════════════════════════
# Baseline 3 — Neyman-Scott parametric sampler
# ══════════════════════════════════════════════════════════════════════════════

class NeymanScottBaseline(_BaselineEvaluator):
    """
    Independent parametric monthly flood generator.

    Fits marginal distributions to the training record (pooled across all
    months) for each site independently:
      Frequency : Poisson(lambda)
      Intensity : Exponential(scale) — fit to positive intensity values
      Duration  : Exponential(scale) — fit to positive duration values

    Generates T months of synthetic flood statistics by independent sampling
    from the fitted distributions. No temporal dependence, no signal
    conditioning, no joint coherence between variables.

    This is the simplest parametric baseline — it tests whether a model
    that correctly captures marginal distributions but ignores all dependence
    structure can match the observed data.
    """

    def __init__(self, seed: Optional[int] = 42):
        self.rng     = np.random.default_rng(seed)
        self._fitted = False

    def fit(
        self,
        monthly_max_train: pd.DataFrame,
        events_train:      pd.DataFrame,
        thr_map:           Dict[str, float],
    ) -> "NeymanScottBaseline":
        """
        Parameters
        ----------
        monthly_max_train : (N_train, n_sites) — used only for site list
        events_train      : long-format events from extract_events_monthly
        thr_map           : not used directly but stored for consistency
        """
        self.thr_map = thr_map
        self.sites   = list(monthly_max_train.columns)
        self.params  = {}

        print("Fitting NS distributions per site ...")
        for site in self.sites:
            site_ev = events_train[events_train["Site"] == site]
            if site_ev.empty:
                continue

            freq_vals = site_ev["Frequency"].dropna().values
            int_vals  = site_ev["Intensity"].dropna()
            dur_vals  = site_ev["Duration"].dropna()

            # Poisson: lambda = mean frequency
            lam = float(np.mean(freq_vals)) if len(freq_vals) > 0 else 1.0

            # Exponential: fit to positive values only (MLE = mean)
            int_pos = int_vals[int_vals > 0].values
            dur_pos = dur_vals[dur_vals > 0].values
            int_scale = float(np.mean(int_pos)) if len(int_pos) > 0 else 1.0
            dur_scale = float(np.mean(dur_pos)) if len(dur_pos) > 0 else 1.0

            # fraction of months with at least one event
            p_event = float((freq_vals > 0).mean()) if len(freq_vals) > 0 else 0.5

            self.params[site] = {
                "lambda":    lam,
                "int_scale": int_scale,
                "dur_scale": dur_scale,
                "p_event":   p_event,
            }

        self._fitted = True
        print(f"NS fitted for {len(self.params)} sites.")
        return self

    def generate(
        self,
        T:      int,
        n_sims: int = 1000,
    ) -> Dict[str, np.ndarray]:
        """
        Generate n_sims sequences of length T months by independent sampling.
        Each month and each sim is an independent draw from fitted distributions.
        """
        assert self._fitted
        results = {}
        n_sites = len(self.sites)

        for s_idx, site in enumerate(self.sites):
            print(f"  [{s_idx+1:>3}/{n_sites}] {site[:60]}", end="\r")
            p = self.params.get(site)
            if p is None:
                continue

            ens = np.zeros((T, n_sims, 3))

            # Frequency: Poisson — shape (T, n_sims)
            freq = self.rng.poisson(p["lambda"], size=(T, n_sims)).astype(float)

            # Intensity: Exponential — shape (T, n_sims)
            # set to 0 where frequency == 0
            intensity = self.rng.exponential(p["int_scale"], size=(T, n_sims))
            intensity[freq == 0] = 0.0

            # Duration: Exponential — shape (T, n_sims)
            duration  = self.rng.exponential(p["dur_scale"], size=(T, n_sims))
            duration[freq == 0] = 0.0

            ens[:, :, 0] = freq
            ens[:, :, 1] = intensity
            ens[:, :, 2] = duration

            results[site] = ens

        print(f"\nNS done. {n_sims}×{T}×{n_sites} ensemble generated.")
        return results
