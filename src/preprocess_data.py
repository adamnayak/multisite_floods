from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pathlib import Path


try:
    from scipy.stats import gumbel_r
except Exception:
    gumbel_r = None


# ----------------------------
# Helper functions (pure)
# ----------------------------

def _validate_series_window(
    s: pd.Series,
    requested_start: str,
    requested_end: str,
    tolerance_days: int = 366,
) -> Tuple[bool, str]:
    if s is None or s.dropna().empty:
        return False, "empty series"

    req_s = pd.Timestamp(requested_start)
    req_e = pd.Timestamp(requested_end)
    act_s = s.first_valid_index()
    act_e = s.last_valid_index()

    if act_s is None or act_e is None:
        return False, "no valid indices"

    if act_s > req_s + pd.Timedelta(days=tolerance_days):
        return False, f"starts too late: actual {act_s.date()} vs requested {req_s.date()}"

    if act_e < req_e - pd.Timedelta(days=tolerance_days):
        return False, f"ends too early: actual {act_e.date()} vs requested {req_e.date()}"

    return True, "ok"

def _ensure_dtindex(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_convert(None)
    return df.sort_index()


def compute_threshold_map(
    Q_daily: pd.DataFrame,
    mode: str = "quantile",
    q: float = 0.9,
    scalar_threshold: Optional[float] = None,
) -> Dict[str, float]:
    """
    Returns per-gauge threshold map.

    mode:
      - "quantile": per-gauge quantile threshold
      - "scalar": same threshold for all gauges
    """
    if mode == "quantile":
        return {c: float(Q_daily[c].dropna().quantile(q)) for c in Q_daily.columns}
    elif mode == "scalar":
        if scalar_threshold is None:
            raise ValueError("scalar_threshold must be provided when mode='scalar'.")
        return {c: float(scalar_threshold) for c in Q_daily.columns}
    else:
        raise ValueError(f"Unknown threshold mode: {mode}")


def extract_gauge_events(
    Q_daily: pd.DataFrame,
    threshold_map: Dict[str, float],
    min_duration_days: int = 1,
    gap_merge_days: int = 0,
) -> pd.DataFrame:
    """
    Extract exceedance events per gauge.

    An event is a run of days where Q > threshold, optionally merging runs separated by <= gap_merge_days.
    Returns a tidy event table with: gauge, start, end, peak, duration_days.
    """
    Q_daily = _ensure_dtindex(Q_daily)
    events = []

    for gauge in Q_daily.columns:
        thr = threshold_map[gauge]
        s = Q_daily[gauge].astype(float)

        is_exc = (s > thr) & s.notna()
        if not is_exc.any():
            continue

        dates = s.index

        # Identify runs in boolean array
        exc = is_exc.values.astype(int)
        # run starts where exc goes 0->1, ends where 1->0
        starts = np.where((exc[1:] == 1) & (exc[:-1] == 0))[0] + 1
        ends = np.where((exc[1:] == 0) & (exc[:-1] == 1))[0]

        # If starts/ends at boundaries
        if exc[0] == 1:
            starts = np.r_[0, starts]
        if exc[-1] == 1:
            ends = np.r_[ends, len(exc) - 1]

        # Build initial event intervals
        intervals = [(int(st), int(en)) for st, en in zip(starts, ends)]

        # Optional merge small gaps
        if gap_merge_days > 0 and len(intervals) > 1:
            merged = [intervals[0]]
            for st, en in intervals[1:]:
                prev_st, prev_en = merged[-1]
                gap = (st - prev_en - 1)
                if gap <= gap_merge_days:
                    merged[-1] = (prev_st, en)
                else:
                    merged.append((st, en))
            intervals = merged

        # Filter by min duration and compute stats
        for st, en in intervals:
            start_dt = dates[st]
            end_dt = dates[en]
            duration = (end_dt - start_dt).days + 1
            if duration < min_duration_days:
                continue

            peak = float(s.iloc[st:en+1].max())
            events.append(
                {
                    "gauge": gauge,
                    "start": start_dt,
                    "end": end_dt,
                    "peak": peak,
                    "duration_days": int(duration),
                }
            )

    return pd.DataFrame(events)


def events_to_monthly_marks_all_months(
    events: pd.DataFrame,
    months_index: pd.DatetimeIndex,
    gauges: List[str],
) -> pd.DataFrame:
    """
    Aggregate events to monthly marks:
      frequency      = # events (by event start month)
      max_intensity  = max peak among events in month
      duration_days  = max duration among events in month

    Critically: returns ALL months in months_index and fills zeros where no events.
    """
    # Base output with all months
    out_cols = {}
    n_months = len(months_index)
    for g in gauges:
        out_cols[f"{g}_frequency"] = np.zeros(n_months, dtype=int)
        out_cols[f"{g}_max_intensity"] = np.zeros(n_months, dtype=float)
        out_cols[f"{g}_duration_days"] = np.zeros(n_months, dtype=int)
    
    out = pd.DataFrame(out_cols, index=months_index)


    if events is None or len(events) == 0:
        return out

    e = events.copy()
    e["month"] = e["start"].dt.to_period("M").dt.to_timestamp(how="start")

    # Group per gauge-month
    grp = e.groupby(["gauge", "month"], as_index=False).agg(
        frequency=("peak", "size"),
        max_intensity=("peak", "max"),
        duration_days=("duration_days", "max"),
    )

    # Write into output (only where months are in index)
    for _, row in grp.iterrows():
        g = row["gauge"]
        m = row["month"]
        if m in out.index:
            out.loc[m, f"{g}_frequency"] = int(row["frequency"])
            out.loc[m, f"{g}_max_intensity"] = float(row["max_intensity"])
            out.loc[m, f"{g}_duration_days"] = int(row["duration_days"])

    return out

def empirical_return_levels(
    df: pd.DataFrame,
    cols: List[str],
    targets: Tuple[float, ...] = (10, 100),
    resample_rule: str = "YE",
) -> pd.DataFrame:
    """
    Empirical return-period levels from a (typically monthly) series via annual maxima.

    Returns a DataFrame indexed by col name, with columns T{target}.
    """
    results = {}
    t_arr = np.asarray(targets, dtype=float)

    for c in cols:
        ams = df[c].resample(resample_rule).max().dropna()
        n = len(ams)
        if n < 2:
            raise ValueError(f"Series '{c}' has fewer than 2 annual maxima.")

        ams_sorted = ams.sort_values(ascending=False).reset_index(drop=True)
        rank = np.arange(1, n + 1)
        exceed_prob = rank / (n + 1)
        return_period = 1.0 / exceed_prob  # years

        # Interpolate in RP space (stable enough for typical use)
        rl = np.interp(t_arr, return_period[::-1], ams_sorted[::-1])
        results[c] = rl

    col_names = [f"T{int(t)}" for t in targets]
    return pd.DataFrame(results, index=col_names).T


def get_valid_date_range(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """First/last non-NA timestamps per column."""
    first_valid_dates = df.apply(lambda col: col.first_valid_index())
    last_valid_dates = df.apply(lambda col: col.last_valid_index())
    return first_valid_dates, last_valid_dates


def _ensure_naive_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    idx = df.index
    if isinstance(idx, pd.DatetimeIndex) and idx.tz is not None:
        df.index = idx.tz_convert(None)
    return df


def _to_month_end_index(df: pd.DataFrame) -> pd.DataFrame:
    """Convert DatetimeIndex to month-end timestamps (ME) consistently."""
    out = df.copy()
    out.index = pd.to_datetime(out.index)
    # If already month end, this is stable; if not, resampling will define it anyway.
    return out

def rl_df_to_threshold_map(
    rl_df: pd.DataFrame,
    rp: Union[int, str] = 2,
    prefix: str = "T",
) -> Dict[str, float]:
    """
    Convert a return-level DataFrame (index=gauge, columns like 'T2','T5',...)
    into a threshold map {gauge: threshold} for a chosen return period.
    """
    col = rp if isinstance(rp, str) else f"{prefix}{int(rp)}"
    if col not in rl_df.columns:
        raise KeyError(f"Return-level column '{col}' not found. Available: {list(rl_df.columns)}")
    return rl_df[col].dropna().astype(float).to_dict()


def historic_events_to_export(df: pd.DataFrame, sim_id: int = 0) -> pd.DataFrame:
    """
    Convert a historic event dataframe with columns:
      gauge, start, end, peak, duration_days

    into the export format:
      Sim, Month, Storm_Index, Event_ID, Site, Signal, Frequency, Intensity, Duration

    - Month: month of `start` (normalized to month-start timestamp)
    - Frequency: count of events per (Site, Month), repeated on each row
    - Storm_Index: enumerated per row (1..N)
    - Event_ID: f"{Sim}_{Storm_Index}"
    - Signal: set to NaN (not available)
    """

    df = df.copy()

    # Ensure datetime
    df["start"] = pd.to_datetime(df["start"])

    # Month = normalized to month-start (consistent with your other export)
    df["Month"] = df["start"].dt.to_period("M").dt.to_timestamp()

    # Site
    df["Site"] = df["gauge"]

    # Sim (constant for all obs)
    df["Sim"] = int(sim_id)

    # Storm_Index: simple enumeration
    df = df.reset_index(drop=True)
    df["Storm_Index"] = np.arange(1, len(df) + 1)

    # Event_ID: Sim_StormIndex (you can change this pattern if needed)
    df["Event_ID"] = df["Sim"].astype(str) + "_" + df["Storm_Index"].astype(str)

    # Frequency: count of events per Site-Month
    df["Frequency"] = (
        df.groupby(["Site", "Month"])["Storm_Index"]
          .transform("size")
          .astype(float)   # keep numeric
    )

    # Intensity & Duration
    df["Intensity"] = df["peak"]
    df["Duration"] = df["duration_days"]

    # Signal not defined here
    df["Signal"] = np.nan

    # Reorder columns to match export format
    out = df[[
        "Sim", "Month", "Storm_Index", "Event_ID",
        "Site", "Signal", "Frequency", "Intensity", "Duration"
    ]]

    return out


def events_to_export(
    df: pd.DataFrame,
    signal_suffix: str | None = None,
    ds=None,
    date_col: str = "Unnamed: 0",
    default_sim_id: int = 0,
    keep_zeros: bool = True,
) -> pd.DataFrame:
    """
    General export formatter for:
      1) Copula/expanded sims with (sim_id, t, event_id)
      2) Monthly covariate tables with a date column or DatetimeIndex

    Output columns:
      Sim, Month, Storm_Index, Event_ID, Site, Signal, Frequency, Intensity, Duration

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe (wide, per-month or per-step).
    signal_suffix : str or None
        Suffix after site name for the signal column, e.g.:
          - "forecast_signal"  -> {site}_forecast_signal
          - "max_z_wave"       -> {site}_max_z_wave
        If None, Signal is filled with NaN.
    ds : object, optional
        Object with `monthly_max.index` used to map t -> Month when
        sim_id/t/event_id are present. Required in that case unless
        df already has a usable Month column.
    date_col : str, default "Unnamed: 0"
        Column name to use as the date when in date mode.
    default_sim_id : int, default 0
        Sim id to use when no sim_id column exists (date mode).
    keep_zeros : bool, default True
        If False, drop rows with Frequency <= 0.

    Modes
    -----
    * Sim-mode: if df has columns sim_id, t, event_id (or they are in a MultiIndex).
      - Month is computed from t and ds.monthly_max.index[-1]
      - Sim = sim_id
      - Storm_Index = event_id
      - Event_ID = "{sim_id}_{t}_{event_id}"

    * Date-mode: if df has a date_col or Datetime/Period index (and no sim_id/t/event_id).
      - Month from date
      - Sim = default_sim_id
      - Storm_Index = 1..N (row-wise enumeration)
      - Event_ID = "{Sim}_{Storm_Index}"
    """

    # Work on a copy
    df = df.copy()

    # ---- Detect sim-mode vs date-mode ----
    sim_cols = ["sim_id", "t", "event_id"]
    has_sim_cols = all(c in df.columns for c in sim_cols)

    # If MultiIndex looks like (sim_id, t, event_id), promote to columns
    if isinstance(df.index, pd.MultiIndex) and not has_sim_cols:
        idx_names = set(df.index.names or [])
        if set(sim_cols).issubset(idx_names):
            df = df.reset_index()
            has_sim_cols = True

    has_datetime_index = isinstance(df.index, (pd.DatetimeIndex, pd.PeriodIndex))
    has_date_col = date_col in df.columns

    if has_sim_cols:
        mode = "sim"
    elif has_date_col or has_datetime_index:
        mode = "date"
    else:
        raise ValueError(
            "Cannot infer mode: need sim_id/t/event_id (sim-mode) or "
            f"a date column '{date_col}' or Datetime/Period index (date-mode)."
        )

    # ---- Build Month, Sim, Storm_Index, Event_ID ----

    def _clean_intish(x):
        if pd.isna(x):
            return np.nan
        try:
            xf = float(x)
            return int(xf) if xf.is_integer() else x
        except Exception:
            return x

    if mode == "sim":
        # Required id columns must exist now
        if not all(c in df.columns for c in sim_cols):
            raise ValueError("sim-mode requires columns: sim_id, t, event_id")

        # Month mapping from t (1-indexed) using ds.monthly_max.index[-1]
        if "Month" not in df.columns:
            if ds is None:
                raise ValueError(
                    "In sim-mode, either provide `ds` with monthly_max.index "
                    "or pre-populate a 'Month' column in df."
                )
            base = pd.Timestamp(ds.monthly_max.index[-1])
            base_period = base.to_period("M")
            offsets = df["t"].astype(int) + 1
            df["Month"] = pd.PeriodIndex(
                [base_period + int(k) for k in offsets],
                freq="M",
            ).to_timestamp()

        df["Sim"] = df["sim_id"].astype(int)
        df["Storm_Index"] = df["event_id"]

        t_clean = df["t"].map(_clean_intish)
        e_clean = df["event_id"].map(_clean_intish)

        df["Event_ID"] = (
            df["Sim"].astype(str)
            + "_"
            + t_clean.astype(str)
            + "_"
            + e_clean.astype(str)
        )

    else:  # mode == "date"
        # Month from date column or index
        if has_date_col:
            df["Month"] = pd.to_datetime(df[date_col])
        elif has_datetime_index:
            if isinstance(df.index, pd.PeriodIndex):
                df["Month"] = df.index.to_timestamp()
            else:
                df["Month"] = df.index
        else:
            raise ValueError(
                "Date-mode selected, but neither date_col nor a datetime index is usable."
            )

        df = df.reset_index(drop=True)
        df["Sim"] = int(default_sim_id)
        df["Storm_Index"] = np.arange(1, len(df) + 1, dtype=int)
        df["Event_ID"] = df["Sim"].astype(str) + "_" + df["Storm_Index"].astype(str)

    # ---- Detect sites from *_frequency columns ----
    site_prefixes = sorted(
        {c.rsplit("_", 1)[0] for c in df.columns if c.endswith("_frequency")}
    )

    if not site_prefixes:
        raise ValueError(
            "No sites found from *_frequency columns. "
            "Expected columns like '{site}_frequency'."
        )

    long_parts = []

    for site in site_prefixes:
        tmp = df[["Sim", "Month", "Storm_Index", "Event_ID"]].copy()
        tmp["Site"] = site

        # Frequency, Intensity, Duration
        freq_col = f"{site}_frequency"
        int_col = f"{site}_max_intensity"
        dur_col = f"{site}_duration_days"

        tmp["Frequency"] = df[freq_col] if freq_col in df.columns else np.nan
        tmp["Intensity"] = df[int_col] if int_col in df.columns else np.nan
        tmp["Duration"] = df[dur_col] if dur_col in df.columns else np.nan

        # Signal (optional)
        if signal_suffix is not None:
            sig_col = f"{site}_{signal_suffix}"
            tmp["Signal"] = df[sig_col] if sig_col in df.columns else np.nan
        else:
            tmp["Signal"] = np.nan

        long_parts.append(tmp)

    out = pd.concat(long_parts, ignore_index=True)

    # Reorder columns
    out = out[
        [
            "Sim",
            "Month",
            "Storm_Index",
            "Event_ID",
            "Site",
            "Signal",
            "Frequency",
            "Intensity",
            "Duration",
        ]
    ]

    if not keep_zeros:
        out = out[out["Frequency"].fillna(0) > 0].copy()

    return out


# ----------------------------
# Optional NWIS fetcher
# ----------------------------

def fetch_daily_q_nwis_chunked(
    site_no: str,
    start: str,
    end: str,
    parameter_code: str = "00060",
    chunk_years: int = 20,
) -> pd.Series:
    """
    Fetch NWIS daily values in chunks to avoid silent truncation on long requests.
    """
    import dataretrieval.nwis as nwis

    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)

    parts = []
    cur = start_ts

    while cur <= end_ts:
        nxt = min(cur + pd.DateOffset(years=chunk_years) - pd.Timedelta(days=1), end_ts)

        out = nwis.get_record(
            sites=site_no,
            service="dv",
            start=cur.strftime("%Y-%m-%d"),
            end=nxt.strftime("%Y-%m-%d"),
            parameterCd=parameter_code,
        )
        df = out[0] if isinstance(out, tuple) else out
        if df is None or len(df) == 0:
            cur = nxt + pd.Timedelta(days=1)
            continue

        if "datetime" in df.columns:
            df = df.set_index("datetime")
        df.index = pd.to_datetime(df.index)

        qcols = [c for c in df.columns if str(c).startswith(parameter_code) and not str(c).endswith("_cd")]
        if not qcols:
            qcols = list(df.select_dtypes(include=[np.number]).columns)
        if qcols:
            parts.append(df[qcols[0]].astype(float))

        cur = nxt + pd.Timedelta(days=1)

    if not parts:
        return pd.Series(dtype=float)

    s = pd.concat(parts).sort_index()
    s = s[~s.index.duplicated(keep="first")]
    return s


def fetch_daily_q_nwis(
    site_no: str,
    start: str,
    end: str,
    parameter_code: str = "00060",
) -> pd.Series:
    """
    Fetch NWIS daily values via dataretrieval. Returns a Series.
    Requires: pip install dataretrieval
    """
    import dataretrieval.nwis as nwis  # local import so class can exist without it

    out = nwis.get_record(
        sites=site_no,
        service="dv",
        start=start,
        end=end,
        parameterCd=parameter_code,
    )
    df = out[0] if isinstance(out, tuple) else out
    if df is None or len(df) == 0:
        return pd.Series(dtype=float)

    if "datetime" in df.columns:
        df = df.set_index("datetime")
    df.index = pd.to_datetime(df.index)

    qcols = [c for c in df.columns if str(c).startswith(parameter_code) and not str(c).endswith("_cd")]
    if not qcols:
        qcols = list(df.select_dtypes(include=[np.number]).columns)
    if not qcols:
        return pd.Series(dtype=float)

    return df[qcols[0]].astype(float).sort_index()


# ----------------------------
# Configuration dataclasses
# ----------------------------

@dataclass(frozen=True)
class GaugeSpec:
    name: str
    site_no: str
    start: str
    end: str


@dataclass
class Standardizer:
    means: pd.Series
    stds: pd.Series

    def transform(self, df: pd.DataFrame, cols: Optional[List[str]] = None) -> pd.DataFrame:
        cols = cols or list(self.means.index)
        out = df.copy()
        out[cols] = (out[cols] - self.means[cols]) / self.stds[cols]
        return out

    def inverse_transform(self, df: pd.DataFrame, cols: Optional[List[str]] = None) -> pd.DataFrame:
        cols = cols or list(self.means.index)
        out = df.copy()
        out[cols] = out[cols] * self.stds[cols] + self.means[cols]
        return out


# ----------------------------
# Main class
# ----------------------------

class HydroClimateDataset:
    """
    Loads + preprocesses discharge + climate indices, aligns by valid overlap,
    builds monthly/annual summaries, standardizes, and computes return levels.

    Key outputs after calling .build():
        - self.merged_daily: merged daily dataframe (climate + discharge)
        - self.monthly_max:  monthly maxima (all cols) + monthly discharge sums (standardized)
        - self.annual_max:   annual maxima (all cols)
        - self.standardizer: Standardizer for selected columns (typically discharge sums)
        - self.return_levels_empirical: empirical RL table
        - self.return_levels_gumbel: optional gumbel RL dict
    """

    def __init__(
        self,
        gauges: List[GaugeSpec],
        parameter_code: str = "00060",
        discharge_source: str = "local",  # "local" or "nwis"
        discharge_local_path: Optional[str] = None,  # CSV like Discharge_Preprocess.csv
        discharge_datetime_col: str = "datetime",
        climate_source: Union[str, pd.DataFrame, Callable[[], pd.DataFrame], None] = None,
        climate_indices: Optional[List[str]] = None,
        climate_datetime_col: [int] = 0,
        resample_month: str = "ME",
        resample_year: str = "YE",
        annual_max_rule_for_rl: str = "YE",
        standardize_cols: Optional[List[str]] = None,
        drop_zero_frequency_months: bool = False,
        zero_frequency_cols: Optional[List[str]] = None,
        verbose: bool = True,
    ):
        self.gauges = gauges
        self.parameter_code = parameter_code
        self.discharge_source = discharge_source
        self.discharge_local_path = discharge_local_path
        self.discharge_datetime_col = discharge_datetime_col

        self.climate_source = climate_source
        self.climate_indices = climate_indices
        self.climate_datetime_col = climate_datetime_col
        self.resample_month = resample_month
        self.resample_year = resample_year
        self.annual_max_rule_for_rl = annual_max_rule_for_rl

        self.standardize_cols = standardize_cols  # if None, we’ll infer later
        self.drop_zero_frequency_months = drop_zero_frequency_months
        self.zero_frequency_cols = zero_frequency_cols

        self.verbose = verbose
        self.threshold_mode = "quantile"   # or "scalar"
        self.threshold_q = 0.9
        self.scalar_threshold = None
        
        self.min_duration_days = 1
        self.gap_merge_days = 0
        
        self.event_threshold_map = None
        self.events_table = None
        self.monthly_event_marks = None

        # Artifacts
        self.climate_df: Optional[pd.DataFrame] = None
        self.discharge_daily: Optional[pd.DataFrame] = None
        self.merged_daily: Optional[pd.DataFrame] = None

        self.monthly_max: Optional[pd.DataFrame] = None
        self.annual_max: Optional[pd.DataFrame] = None

        self.standardizer: Optional[Standardizer] = None
        self.return_levels_empirical: Optional[pd.DataFrame] = None
        self.return_levels_gumbel: Optional[Dict[str, Dict[int, float]]] = None

        self.overlap_start: Optional[pd.Timestamp] = None
        self.overlap_end: Optional[pd.Timestamp] = None

    # ----------------------------
    # Loaders
    # ----------------------------

    def load_climate(self) -> pd.DataFrame:
        if self.climate_source is None:
            raise ValueError("climate_source is None. Provide a DataFrame or a callable returning a DataFrame.")

        src = self.climate_source

        # --- Case A: DataFrame passed directly (back-compat) ---
        if isinstance(src, pd.DataFrame):
            df = src.copy()
    
        # --- Case B: callable passed directly (back-compat) ---
        elif callable(src):
            df = src()
    
        # --- Case C: path-like (string or Path) ---
        elif isinstance(src, (str, Path)):
            path = Path(src)
            if not path.exists():
                raise FileNotFoundError(f"Climate file not found: {path}")
    
            suffix = path.suffix.lower()
    
            if suffix == ".csv":
                # datetime column specified by name
                if isinstance(self.climate_datetime_col, str):
                    df = pd.read_csv(
                        path,
                        parse_dates=[self.climate_datetime_col],
                        index_col=self.climate_datetime_col,
                    )
            
                # datetime column specified by integer position
                elif isinstance(self.climate_datetime_col, int):
                    df = pd.read_csv(path)
                    dt_col = df.columns[self.climate_datetime_col]
                    df[dt_col] = pd.to_datetime(df[dt_col])
                    df = df.set_index(dt_col)
            
                else:
                    raise TypeError(
                        "climate_datetime_col must be a column name (str) or column index (int)"
                    )
        
            elif suffix in {".parquet", ".pq"}:
                df = pd.read_parquet(path)
    
            else:
                raise ValueError(
                    f"Unsupported climate file type '{suffix}'. "
                    "Supported: .csv, .parquet"
                )
    
        else:
            raise TypeError(
                "climate_source must be a DataFrame, callable, or path-like object."
            )
    
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("Climate dataframe must have a DatetimeIndex.")

        df = _ensure_naive_datetime_index(df)
        df = df.apply(pd.to_numeric, errors="coerce")

        if self.climate_indices is not None:
            missing = [c for c in self.climate_indices if c not in df.columns]
            if missing:
                raise KeyError(f"Climate indices not found: {missing}")
            df = df[self.climate_indices]

        self.climate_df = df.sort_index()
        return self.climate_df

    def load_discharge(self) -> pd.DataFrame:
        if self.discharge_source not in {"local", "nwis"}:
            raise ValueError("discharge_source must be 'local' or 'nwis'.")

        if self.discharge_source == "local":
            if not self.discharge_local_path:
                raise ValueError("discharge_local_path must be provided when discharge_source='local'.")

            df = pd.read_csv(
                self.discharge_local_path,
                parse_dates=[self.discharge_datetime_col],
            )
            df = df.set_index(self.discharge_datetime_col).sort_index()
            df = df.apply(pd.to_numeric, errors="coerce")
            df = _ensure_naive_datetime_index(df)

            # Optionally restrict to gauge columns only (if your CSV has extras)
            gauge_names = [g.name for g in self.gauges]
            present = [c for c in gauge_names if c in df.columns]
            if present:
                df = df[present]

            self.discharge_daily = df
            return self.discharge_daily

        # NWIS
        series = {}
        for g in self.gauges:
            if self.verbose:
                print(f"Fetching NWIS DV for {g.name} ({g.site_no}) {g.start} → {g.end}")
            s = fetch_daily_q_nwis_chunked(g.site_no, g.start, g.end, self.parameter_code, chunk_years=20)
            series[g.name] = s

        df = pd.concat(series, axis=1).sort_index()
        df = df.apply(pd.to_numeric, errors="coerce")
        df = _ensure_naive_datetime_index(df)

        self.discharge_daily = df
        return self.discharge_daily

    # ----------------------------
    # Alignment + summaries
    # ----------------------------

    def compute_overlap(self, climate: pd.DataFrame, discharge: pd.DataFrame) -> Tuple[pd.Timestamp, pd.Timestamp]:
        fc, lc = get_valid_date_range(climate)
        fd, ld = get_valid_date_range(discharge)

        start = max(fc.max(), fd.max())
        end = min(lc.min(), ld.min())

        if pd.isna(start) or pd.isna(end) or start >= end:
            raise ValueError(f"No valid overlap found. start={start}, end={end}")

        self.overlap_start, self.overlap_end = start, end
        return start, end

    def print_date_ranges(self, climate: pd.DataFrame, discharge: pd.DataFrame) -> None:
        fc, lc = get_valid_date_range(climate)
        fd, ld = get_valid_date_range(discharge)

        print("\nDate ranges (first_valid → last_valid):")
        print("Climate:")
        for c in climate.columns:
            print(f"  {c:>20}: {fc[c]} → {lc[c]}")
        print("Discharge:")
        for c in discharge.columns:
            print(f"  {c:>20}: {fd[c]} → {ld[c]}")

        if self.overlap_start is not None and self.overlap_end is not None:
            print(f"\nOverlap used: {self.overlap_start} → {self.overlap_end}")

    def merge_and_clip(self, climate: pd.DataFrame, discharge: pd.DataFrame) -> pd.DataFrame:
        start, end = self.compute_overlap(climate, discharge)

        climate_f = climate.loc[start:end]
        discharge_f = discharge.loc[start:end]

        merged = climate_f.merge(discharge_f, left_index=True, right_index=True, how="inner")
        merged = merged.sort_index()
        self.merged_daily = merged
        return merged

    def build_monthly_event_marks(
        self,
        Q_daily: pd.DataFrame,
        threshold_map: Optional[Dict[str, float]] = None,
        threshold_mode: str = "quantile",
        q: float = 0.9,
        scalar_threshold: Optional[float] = None,
        min_duration_days: int = 1,
        gap_merge_days: int = 0,
        months_index: Optional[pd.DatetimeIndex] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Build monthly event marks for all months (0-filled).
    
        Per gauge-month:
          - frequency = number of events (by event start month)
          - max_intensity = peak of the max-intensity event
          - duration_days = duration of the max-intensity event
          - mean_intensity_nonmax = mean peak among non-max events (0 if none)
          - mean_duration_nonmax = mean duration among non-max events (0 if none)
    
        Returns
        -------
        (marks, events)
          marks: monthly dataframe (all months included; 0-filled)
          events: event table from extract_gauge_events
        """
        Q_daily = _ensure_dtindex(Q_daily)
        gauges = [g.name for g in self.gauges if g.name in Q_daily.columns]
        Qg = Q_daily[gauges]
    
        # Month index default: full span of Q_daily (month starts)
        if months_index is None:
            months_index = pd.date_range(
                Qg.index.min().to_period("M").to_timestamp(how="start"),
                Qg.index.max().to_period("M").to_timestamp(how="start"),
                freq="MS",
            )
    
        # --- thresholds ---
        if threshold_map is None:
            thr_map = compute_threshold_map(Qg, mode=threshold_mode, q=q, scalar_threshold=scalar_threshold)
        else:
            missing = [g for g in gauges if g not in threshold_map]
            if missing:
                raise KeyError(f"threshold_map missing gauges: {missing}")
            thr_map = {g: float(threshold_map[g]) for g in gauges}
    
        # --- extract events ---
        events = extract_gauge_events(
            Qg,
            threshold_map=thr_map,
            min_duration_days=min_duration_days,
            gap_merge_days=gap_merge_days,
        )
    
        # --- initialize marks (all months, all gauges, 0-filled) ---
        n_months = len(months_index)
        cols_data = {}
    
        for g in gauges:
            cols_data[f"{g}_frequency"] = np.zeros(n_months, dtype=int)
            cols_data[f"{g}_max_intensity"] = np.zeros(n_months, dtype=float)
            cols_data[f"{g}_duration_days"] = np.zeros(n_months, dtype=int)
            cols_data[f"{g}_mean_intensity_nonmax"] = np.zeros(n_months, dtype=float)
            cols_data[f"{g}_mean_duration_nonmax"] = np.zeros(n_months, dtype=float)
    
        marks = pd.DataFrame(cols_data, index=months_index)
    
        # If no events, store + return
        if events is None or len(events) == 0:
            self.event_threshold_map = thr_map
            self.events_table = events
            self.monthly_event_marks = marks
            return marks, events
    
        # --- aggregate events -> monthly marks ---
        e = events.copy()
        e["month"] = e["start"].dt.to_period("M").dt.to_timestamp(how="start")
    
        # group by gauge-month, select max-intensity event + compute nonmax means
        for (gauge, month), df in e.groupby(["gauge", "month"]):
            if month not in marks.index:
                continue
    
            # tie-break: peak desc, duration desc, start asc
            df2 = df.sort_values(["peak", "duration_days", "start"], ascending=[False, False, True])
    
            max_row = df2.iloc[0]
            nonmax = df2.iloc[1:]
    
            freq = int(len(df2))
            max_int = float(max_row["peak"])
            dur_max = int(max_row["duration_days"])
    
            mean_int_nonmax = float(nonmax["peak"].mean()) if len(nonmax) > 0 else 0.0
            mean_dur_nonmax = float(nonmax["duration_days"].mean()) if len(nonmax) > 0 else 0.0
    
            marks.loc[month, f"{gauge}_frequency"] = freq
            marks.loc[month, f"{gauge}_max_intensity"] = max_int
            marks.loc[month, f"{gauge}_duration_days"] = dur_max
            marks.loc[month, f"{gauge}_mean_intensity_nonmax"] = mean_int_nonmax
            marks.loc[month, f"{gauge}_mean_duration_nonmax"] = mean_dur_nonmax
    
        # store
        self.event_threshold_map = thr_map
        self.events_table = events
        self.monthly_event_marks = marks
        return marks, events

    def build_summaries(self, merged_daily: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # Identify discharge gauge columns present
        gauge_cols = [g.name for g in self.gauges]
        gauge_cols = [c for c in gauge_cols if c in merged_daily.columns]
        if not gauge_cols:
            raise ValueError("No gauge discharge columns found in merged_daily.")
    
        # Decide which gauges to standardize
        if self.standardize_cols is None:
            std_gauges = gauge_cols
        else:
            std_gauges = [c for c in self.standardize_cols if c in gauge_cols]
            if not std_gauges:
                raise ValueError("standardize_cols provided, but none were found in discharge gauge columns.")
    
        # Monthly maxima of all merged columns (climate + discharge)
        mon_max_all = merged_daily.resample(self.resample_month).max()
    
        # Annual maxima of all merged columns
        yr_max = merged_daily.resample(self.resample_year).max()
    
        # -------------------------
        # (1) Discharge monthly MAX
        # -------------------------
        mon_gauge_max = merged_daily[gauge_cols].resample(self.resample_month).max()
        mon_gauge_max.columns = [f"{c}_max" for c in mon_gauge_max.columns]
    
        max_cols_to_std = [f"{c}_max" for c in std_gauges]
        max_means = mon_gauge_max[max_cols_to_std].mean()
        max_stds = mon_gauge_max[max_cols_to_std].std(ddof=1).replace(0.0, np.nan)
        self.standardizer_max = Standardizer(means=max_means, stds=max_stds)
    
        mon_gauge_max_z = self.standardizer_max.transform(mon_gauge_max, cols=max_cols_to_std)
        mon_gauge_max_z.columns = [c.replace("_max", "_max_z") for c in mon_gauge_max_z.columns]
    
        # -------------------------
        # (2) Discharge monthly SUM
        # -------------------------
        mon_gauge_sum = merged_daily[gauge_cols].resample(self.resample_month).sum()
        mon_gauge_sum.columns = [f"{c}_sum" for c in mon_gauge_sum.columns]
    
        sum_cols_to_std = [f"{c}_sum" for c in std_gauges]
        sum_means = mon_gauge_sum[sum_cols_to_std].mean()
        sum_stds = mon_gauge_sum[sum_cols_to_std].std(ddof=1).replace(0.0, np.nan)
        self.standardizer_sum = Standardizer(means=sum_means, stds=sum_stds)
    
        mon_gauge_sum_z = self.standardizer_sum.transform(mon_gauge_sum, cols=sum_cols_to_std)
        mon_gauge_sum_z.columns = [c.replace("_sum", "_sum_z") for c in mon_gauge_sum_z.columns]
    
        # -------------------------
        # (3) Assemble monthly table
        # -------------------------
        mon_feat = mon_max_all.copy()
    
        mon_feat = mon_feat.join(mon_gauge_max, how="left")
        mon_feat = mon_feat.join(mon_gauge_max_z, how="left")
        mon_feat = mon_feat.join(mon_gauge_sum, how="left")
        mon_feat = mon_feat.join(mon_gauge_sum_z, how="left")
    
        # Drop raw gauge columns (duplicates of *_max)
        mon_feat = mon_feat.drop(columns=gauge_cols, errors="ignore")
    
        # Ensure consistent index convention (month-end is what you're using elsewhere)
        mon_feat = _to_month_end_index(mon_feat)
    
        # Add month cyclic features
        mon_feat["month"] = mon_feat.index.month
        mon_feat["sin_month"] = np.sin(2 * np.pi * mon_feat["month"] / 12.0)
        mon_feat["cos_month"] = np.cos(2 * np.pi * mon_feat["month"] / 12.0)
    
        # -------------------------
        # (4) Build + join event marks (AFTER mon_feat exists)
        # -------------------------
        months_index = pd.date_range(
            mon_feat.index.min().to_period("M").to_timestamp(how="start"),
            mon_feat.index.max().to_period("M").to_timestamp(how="start"),
            freq="MS",
        )
    
        Q_for_events = self.discharge_daily[gauge_cols] if self.discharge_daily is not None else merged_daily[gauge_cols]
    
        marks, events = self.build_monthly_event_marks(
            Q_daily=Q_for_events,
            threshold_map=getattr(self, "event_threshold_map", None),  # if you set it externally, it will use it
            threshold_mode=getattr(self, "threshold_mode", "quantile"),
            q=getattr(self, "threshold_q", 0.9),
            scalar_threshold=getattr(self, "scalar_threshold", None),
            min_duration_days=getattr(self, "min_duration_days", 1),
            gap_merge_days=getattr(self, "gap_merge_days", 0),
            months_index=months_index,
        )
    
        # Convert marks (month-start) -> month-end to match mon_feat
        marks_me = marks.copy()
        marks_me.index = marks_me.index + pd.offsets.MonthEnd(0)
    
        mon_feat = mon_feat.join(marks_me, how="left").fillna(0)
    
        # -------------------------
        # (5) Optional: drop months where some "frequency" is zero
        # -------------------------
        if self.drop_zero_frequency_months:
            zcols = self.zero_frequency_cols or []
            missing = [c for c in zcols if c not in mon_feat.columns]
            if missing:
                raise KeyError(f"zero_frequency_cols not in monthly dataframe: {missing}")
            mask = np.ones(len(mon_feat), dtype=bool)
            for c in zcols:
                mask &= (mon_feat[c] != 0)
            mon_feat = mon_feat.loc[mask]
    
        self.monthly_max = mon_feat
        self.annual_max = yr_max
        return mon_feat, yr_max

    # ----------------------------
    # Return levels
    # ----------------------------

    def compute_return_levels(
        self,
        cols: Optional[List[str]] = None,
        targets: Tuple[int, ...] = (2, 5, 10, 100),
        fit_gumbel: bool = True,
        annual_rule: Optional[str] = None,   # override if desired
    ) -> Tuple[pd.DataFrame, Optional[Dict[str, Dict[int, float]]]]:
        """
        Compute return levels from an Annual-Maximum Series (AMS).
    
        Empirical RL: Weibull plotting positions + interpolation.
        Gumbel RL: fit Gumbel to AMS and evaluate quantiles.
    
        Parameters
        ----------
        cols : list[str], optional
            Discharge gauge columns to use (e.g., ["st_louis", ...]).
            If None, uses gauge names from self.gauges that exist in data.
        targets : tuple[int,...]
            Return periods in years.
        fit_gumbel : bool
            Whether to also compute Gumbel-based RLs.
        annual_rule : str, optional
            Pandas resample rule for annual maxima (e.g., "A", "A-SEP").
            Defaults to self.annual_max_rule_for_rl.
        """
        annual_rule = annual_rule or self.annual_max_rule_for_rl
    
        # Prefer raw daily discharge for AMS
        if self.discharge_daily is not None:
            base = self.discharge_daily
        elif self.merged_daily is not None:
            base = self.merged_daily
        else:
            raise RuntimeError("No daily data available. Call build() first.")
    
        # Columns default to gauge names present
        cols = cols or [g.name for g in self.gauges if g.name in base.columns]
        if not cols:
            raise ValueError("No requested discharge columns found in daily data for return-level calculation.")
    
        # --- Build AMS explicitly from daily series ---
        ams_df = base[cols].resample(annual_rule).max().dropna(how="all")
    
        # --- Empirical RL from AMS (no extra resampling inside) ---
        # We'll reuse your empirical_return_levels by giving it AMS and resample_rule=None-equivalent.
        # simplest: compute directly here from ams_df column-by-column.
        results = {}
        t_arr = np.asarray(targets, dtype=float)
    
        for c in cols:
            ams = ams_df[c].dropna()
            n = len(ams)
            if n < 2:
                raise ValueError(f"Series '{c}' has fewer than 2 annual maxima.")
    
            ams_sorted = ams.sort_values(ascending=False).to_numpy()
            rank = np.arange(1, n + 1)
            exceed_prob = rank / (n + 1)
            return_period = 1.0 / exceed_prob
    
            rl = np.interp(t_arr, return_period[::-1], ams_sorted[::-1])
            results[c] = rl
    
        rl_emp = pd.DataFrame(results, index=[f"T{int(t)}" for t in targets]).T
        self.return_levels_empirical = rl_emp
    
        # --- Optional Gumbel RL ---
        rl_gum = None
        if fit_gumbel:
            if gumbel_r is None:
                raise ImportError("scipy not available (needed for Gumbel). Install scipy or set fit_gumbel=False.")
    
            rl_gum = {}
            aep_probs = [1 - 1 / rp for rp in targets]  # non-exceedance probs
    
            for c in cols:
                data = ams_df[c].dropna().values
                if len(data) < 2:
                    continue
                loc, scale = gumbel_r.fit(data)
                q = gumbel_r.ppf(aep_probs, loc=loc, scale=scale)
                rl_gum[c] = {int(rp): float(val) for rp, val in zip(targets, q)}
    
            self.return_levels_gumbel = rl_gum
    
        return rl_emp, rl_gum

    def threshold_map_from_rl(self, rl_df: pd.DataFrame, rp: int) -> Dict[str, float]:
        return rl_df_to_threshold_map(rl_df, rp=rp)

    # ----------------------------
    # Orchestration
    # ----------------------------

    def build(self) -> "HydroClimateDataset":
        climate = self.load_climate()
        discharge = self.load_discharge()
        print("DISCHARGE COLS:", list(discharge.columns)[:20])
        print("GAUGE NAMES:", [g.name for g in self.gauges][:20])
        print("Intersection:", set(discharge.columns) & set(g.name for g in self.gauges))

        merged = self.merge_and_clip(climate, discharge)
        if self.verbose:
            self.print_date_ranges(climate, discharge)

        self.build_summaries(merged)
        return self

# ----------------------------
# Split Class
# ----------------------------

@dataclass
class CVSplitConfig:
    # File + parsing
    csv_path: str = "Discharge_Preprocess.csv"
    out_dir: str = "CV_Data_Splits"
    date_col: str = "date"
    date_format: str | None = None

    # 1) Consecutive splits
    consecutive_train_years: float = 25.0
    consecutive_test_years: float = 2
    n_consecutive_splits: int = 3

    # 2) Building block splits
    building_block_specs: Optional[List[Tuple[float, float]]] = None  # list of (train_years, test_years)

    # 3) Standard split
    standard_spec: Tuple[float, float] = (81.0, 8.0)


@dataclass
class TimeSeriesCVSplitter:
    cfg: CVSplitConfig
    df_: pd.DataFrame = field(default=None, init=False, repr=False)
    start_date_: pd.Timestamp = field(default=None, init=False)
    end_date_: pd.Timestamp = field(default=None, init=False)

    def load(self) -> "TimeSeriesCVSplitter":
        df = pd.read_csv(self.cfg.csv_path)

        if self.cfg.date_col not in df.columns:
            raise ValueError(
                f"Column '{self.cfg.date_col}' not found in CSV. "
                f"Available columns: {df.columns.tolist()}"
            )

        # Parse and sort by date
        df[self.cfg.date_col] = pd.to_datetime(df[self.cfg.date_col], format=self.cfg.date_format)
        df = df.sort_values(self.cfg.date_col)

        # Original overall range (before trimming)
        orig_start = df[self.cfg.date_col].min()
        orig_end = df[self.cfg.date_col].max()

        # All non-date columns must be present (non-NA)
        data_cols = [c for c in df.columns if c != self.cfg.date_col]
        if not data_cols:
            raise ValueError("No data columns found besides the date column.")

        mask_all_present = df[data_cols].notna().all(axis=1)

        if not mask_all_present.all():
            df_complete = df.loc[mask_all_present].copy()
            if df_complete.empty:
                raise ValueError(
                    "No time period where all columns have non-missing data. "
                    "Check your input file."
                )

            new_start = df_complete[self.cfg.date_col].min()
            new_end = df_complete[self.cfg.date_col].max()

            print(
                "Trimming data to period with complete data in all columns:\n"
                f"  Original range : {orig_start.date()} → {orig_end.date()}\n"
                f"  Complete range : {new_start.date()} → {new_end.date()}"
            )

            df = df_complete
        else:
            # still nice to log the fact that no trimming was needed
            print(
                "All rows have complete data in all columns.\n"
                f"Using full range: {orig_start.date()} → {orig_end.date()}"
            )

        # Now set index to date and store
        df = df.set_index(self.cfg.date_col)

        self.df_ = df
        self.start_date_ = df.index.min()
        self.end_date_ = df.index.max()

        # Fill defaults if needed
        if self.cfg.building_block_specs is None:
            self.cfg.building_block_specs = [
                (50.0, 4.0),
                (60.0, 5.0),
                (70.0, 6.0),
                (80.0, 7.0),
            ]

        return self

    @staticmethod
    def _years_to_years_months(years: float) -> Tuple[int, int]:
        """Convert float years to integer (years, months). E.g. 2.5 -> (2, 6)."""
        whole_years = int(years)
        frac = years - whole_years
        months = int(round(frac * 12))
        if months == 12:
            whole_years += 1
            months = 0
        return whole_years, months

    def _slice_by_offset(
        self,
        start: pd.Timestamp,
        train_years: float,
        test_years: float,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp, pd.Timestamp]:
        """
        Train: [start, train_end]
        Test : [train_end+1d, test_end]
        """
        if self.df_ is None:
            raise RuntimeError("Data not loaded. Call .load() first.")

        train_y, train_m = self._years_to_years_months(train_years)
        test_y, test_m = self._years_to_years_months(test_years)

        train_end = start + pd.DateOffset(years=train_y, months=train_m) - pd.Timedelta(days=1)
        test_start = train_end + pd.Timedelta(days=1)
        test_end = test_start + pd.DateOffset(years=test_y, months=test_m) - pd.Timedelta(days=1)

        train = self.df_.loc[(self.df_.index >= start) & (self.df_.index <= train_end)]
        test = self.df_.loc[(self.df_.index >= test_start) & (self.df_.index <= test_end)]

        return train, test, train_end, test_end

    def _ensure_out_dir(self) -> Path:
        out_dir = Path(self.cfg.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir

    def _save_split(self, train: pd.DataFrame, test: pd.DataFrame, base_name: str) -> None:
        out_dir = self._ensure_out_dir()
        train_path = out_dir / f"{base_name}_train.csv"
        test_path = out_dir / f"{base_name}_test.csv"
        train.to_csv(train_path)
        test.to_csv(test_path)

        # lightweight log
        print(f"{base_name}:")
        if not train.empty:
            print(f"  Train: {train.index.min().date()} → {train.index.max().date()} ({len(train)} days)")
        else:
            print("  Train: EMPTY")
        if not test.empty:
            print(f"  Test : {test.index.min().date()} → {test.index.max().date()} ({len(test)} days)")
        else:
            print("  Test : EMPTY")
        print(f"  Saved: {train_path.name}, {test_path.name}\n")

    def write_all(self) -> None:
        """Generate and save all configured split types."""
        if self.df_ is None:
            self.load()

        print(f"Full data range: {self.start_date_.date()} → {self.end_date_.date()}\n")

        # 1) Consecutive splits
        print("=== Consecutive splits ===")
        for i in range(self.cfg.n_consecutive_splits):
            # Start at beginning + i * consecutive_train_years
            # (This pattern matches your original description.)
            step_y, step_m = self._years_to_years_months(i * self.cfg.consecutive_train_years)
            split_start = self.start_date_ + pd.DateOffset(years=step_y, months=step_m)

            train, test, _, _ = self._slice_by_offset(
                start=split_start,
                train_years=self.cfg.consecutive_train_years,
                test_years=self.cfg.consecutive_test_years,
            )
            self._save_split(train, test, base_name=f"split{i+1}")

        # 2) Building blocks
        print("=== Building block splits ===")
        for i, (train_years, test_years) in enumerate(self.cfg.building_block_specs, start=1):
            train, test, _, _ = self._slice_by_offset(
                start=self.start_date_,
                train_years=train_years,
                test_years=test_years,
            )
            self._save_split(train, test, base_name=f"block{i}")

        # 3) Standard
        print("=== Standard split ===")
        std_train_years, std_test_years = self.cfg.standard_spec
        train, test, _, _ = self._slice_by_offset(
            start=self.start_date_,
            train_years=std_train_years,
            test_years=std_test_years,
        )
        self._save_split(train, test, base_name="std")

