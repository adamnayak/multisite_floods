from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, List, Tuple, Any

import numpy as np
import pandas as pd

from dataretrieval import nwis
import requests
import matplotlib.pyplot as plt

def fetch_iv_window(site_no: str,
                    start: pd.Timestamp,
                    end: pd.Timestamp,
                    parameter_code: str = "00060",
                    tz: str | None = None) -> pd.DataFrame:
    start = pd.to_datetime(start)
    end = pd.to_datetime(end)
    end_plus = end + pd.Timedelta(days=1)

    # (optional but strongly recommended) catch bad requests / empty queries
    try:
        resp = nwis.get_record(
            sites=site_no,
            service="iv",
            parameterCd=parameter_code,
            start=start.strftime("%Y-%m-%d"),
            end=end_plus.strftime("%Y-%m-%d"),
        )
    except Exception:
        return pd.DataFrame()

    df = resp[0] if isinstance(resp, tuple) else resp
    if df is None or len(df) == 0:
        return pd.DataFrame()

    df = df.copy()

    # Ensure datetime index
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.set_index("datetime")
    elif not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # Normalize timezone handling consistently between df.index and bounds
    if tz is not None:
        # Make index tz-aware in tz
        if df.index.tz is None:
            df.index = df.index.tz_localize(tz)
        else:
            df.index = df.index.tz_convert(tz)

    # Now make bounds compatible with df.index tz-awareness
    idx_tz = df.index.tz
    if idx_tz is not None:
        # df is tz-aware -> make start/end_plus tz-aware in same tz
        if start.tzinfo is None:
            start = start.tz_localize(idx_tz)
        else:
            start = start.tz_convert(idx_tz)

        if end_plus.tzinfo is None:
            end_plus = end_plus.tz_localize(idx_tz)
        else:
            end_plus = end_plus.tz_convert(idx_tz)

    else:
        # df is tz-naive -> ensure bounds are tz-naive
        if start.tzinfo is not None:
            start = start.tz_convert(None)
        if end_plus.tzinfo is not None:
            end_plus = end_plus.tz_convert(None)

    # Trim
    return df.loc[(df.index >= start) & (df.index <= end_plus)]

def plot_sim_hydrograph(subdaily_sim: pd.DataFrame, sim_event_id: str):
    d = subdaily_sim[subdaily_sim["sim_event_id"] == sim_event_id].sort_values("t_hours")
    if d.empty:
        raise ValueError(f"No rows found for sim_event_id={sim_event_id}")

    plt.figure()
    plt.plot(d["t_hours"].to_numpy(), d["Q_sim"].to_numpy())
    plt.xlabel("Hours since event start (t_hours)")
    plt.ylabel("Discharge (Q_sim, cfs)")
    plt.title(f"Simulated hydrograph: {sim_event_id}\nTemplate={d['template_event_id'].iloc[0]}")
    plt.show()

@dataclass
class SubdailyRefitterConfig:
    parameter_code: str = "00060"
    buffer_hours: int = 0
    tz: Optional[str] = None
    verbose: bool = True

    # Matching behavior
    match_on: str = "duration_days"     # only duration_days supported
    duration_metric: str = "abs"        # "abs" or "logabs"
    allow_cross_gauge: bool = False

    # Output behavior
    keep_only_positive_frequency: bool = True
    min_points_per_event: int = 5
    normalize_by: str = "peak"          # "peak" only (alias)
    eps_peak: float = 1e-12

    # If True, use event_start/event_end to compute duration if duration_days missing
    compute_duration_if_missing: bool = True


@dataclass
class HydroTemplate:
    """Compact template representation for fast refitting."""
    gauge: str
    event_id: str
    duration_days: float
    peak_obs: float
    rel_time_hours: np.ndarray   # starts at 0
    q_norm: np.ndarray           # peak-normalized, max==1


class SubdailyHydrographRefitter:
    """
    Build a historic subdaily event library (templates) and refit/scale those shapes
    to simulated future events using (duration_days, max_intensity).
    """

    def __init__(
        self,
        site_meta: Dict[str, dict],
        events_df: pd.DataFrame,
        fetch_iv_window: FetchIVFunc,
        config: Optional[SubdailyRefitterConfig] = None,
    ):
        self.site_meta = site_meta
        self.events_df = events_df.copy()
        self.fetch_iv_window = fetch_iv_window
        self._iv_start_cache: dict[str, pd.Timestamp | None] = {}  # gauge -> earliest IV timestamp (approx)
        self.cfg = config or SubdailyRefitterConfig()

        # populated by build_library()
        self.subdaily_raw: Optional[pd.DataFrame] = None
        self.event_library: Optional[pd.DataFrame] = None   # event-level metadata
        self.templates: Dict[str, List[HydroTemplate]] = {} # gauge -> templates (fast)
        self._templates_sorted: Dict[str, Tuple[np.ndarray, List[HydroTemplate]]] = {}

        self._validate_events_df()

    # ---------------------------------------------------------------------
    # Public
    # ---------------------------------------------------------------------
    def _probe_has_iv(self, gauge: str, t: pd.Timestamp, window_days: int = 2) -> bool:
        """
        Returns True if IV data exists for gauge in [t, t+window_days].
        Robust to 400/404 since fetch_iv_window should return empty on failure.
        """
        site_no = self.site_meta[gauge]["site"]
        df = self.fetch_iv_window(
            site_no=site_no,
            start=t,
            end=t + pd.Timedelta(days=window_days),
            parameter_code=self.cfg.parameter_code,
            tz=self.cfg.tz,
        )
        return df is not None and not df.empty

    def discover_iv_start(
        self,
        gauge: str,
        *,
        search_start: str | pd.Timestamp = "1900-01-01",
        search_end: str | pd.Timestamp = "2025-01-01",
        step_years: int = 10,
        refine: bool = True,
    ) -> pd.Timestamp | None:
        """
        Find an approximate earliest timestamp where IV exists for this gauge.
        Uses coarse stepping then optional binary refinement.
        Returns None if no IV found in [search_start, search_end].
        """
        if gauge in self._iv_start_cache:
            return self._iv_start_cache[gauge]

        lo = pd.to_datetime(search_start)
        hi = pd.to_datetime(search_end)

        # Coarse forward stepping
        last_fail = lo
        first_success = None

        t = lo
        while t <= hi:
            if self._probe_has_iv(gauge, t):
                first_success = t
                break
            last_fail = t
            t = t + pd.DateOffset(years=step_years)

        if first_success is None:
            self._iv_start_cache[gauge] = None
            return None

        if not refine:
            self._iv_start_cache[gauge] = first_success
            return first_success

        # Refine between last_fail and first_success (binary search in time)
        left = last_fail
        right = first_success

        # If we succeeded immediately at lo, we can just return lo
        if left == right:
            self._iv_start_cache[gauge] = right
            return right

        # Binary search to ~day-level
        while (right - left) > pd.Timedelta(days=2):
            mid = left + (right - left) / 2
            mid = pd.to_datetime(mid)  # ensure Timestamp
            if self._probe_has_iv(gauge, mid):
                right = mid
            else:
                left = mid

        self._iv_start_cache[gauge] = right
        return right

    def prefilter_events_for_iv(
        self,
        events_df: pd.DataFrame | None = None,
        *,
        search_start: str | pd.Timestamp = "1900-01-01",
        search_end: str | pd.Timestamp = "2025-01-01",
        step_years: int = 10,
        refine: bool = True,
        keep_unknown: bool = False,
    ) -> pd.DataFrame:
        """
        Drop events whose start is earlier than the discovered IV start date for that gauge.
        If no IV is found for a gauge:
          - keep_unknown=False -> drop all events for that gauge
          - keep_unknown=True  -> keep them (but fetch likely returns empty later)
        """
        df = (events_df if events_df is not None else self.events_df).copy()
        df["start"] = pd.to_datetime(df["start"])

        bounds = {}
        for g in sorted(df["gauge"].unique()):
            iv0 = self.discover_iv_start(
                g,
                search_start=search_start,
                search_end=search_end,
                step_years=step_years,
                refine=refine,
            )
            bounds[g] = iv0
            if self.cfg.verbose:
                print(f"[IV start] {g}: {iv0}")

        df["iv_start"] = df["gauge"].map(bounds)

        if keep_unknown:
            # Keep events if we couldn't discover a start; otherwise filter by discovered start
            out = df[(df["iv_start"].isna()) | (df["start"] >= df["iv_start"])].copy()
        else:
            # Drop gauges with unknown IV start (none found) and events before start
            out = df[df["iv_start"].notna() & (df["start"] >= df["iv_start"])].copy()

        out = out.drop(columns=["iv_start"])
        return out
    
    def build_library(self) -> None:
        """
        Fetch subdaily IV for all events in events_df and build:
          - subdaily_raw: all fetched points
          - event_library: event-level meta
          - templates: compact arrays for fast matching/scaling
        """
        subdaily = self._build_subdaily_event_dataset(
            events_df=self.events_df,
            site_meta=self.site_meta,
            parameter_code=self.cfg.parameter_code,
            buffer_hours=self.cfg.buffer_hours,
            tz=self.cfg.tz,
            verbose=self.cfg.verbose,
        )

        if subdaily.empty:
            raise RuntimeError("No subdaily data fetched; library cannot be built.")

        subdaily = subdaily.copy()
        subdaily["timestamp"] = pd.to_datetime(subdaily["timestamp"])
        subdaily["Q"] = pd.to_numeric(subdaily["Q"], errors="coerce")
        subdaily = subdaily.dropna(subset=["Q"])

        # event-level stats computed from fetched series
        grp = subdaily.groupby(["gauge", "event_id"], sort=False)
        meta = grp.agg(
            event_peak_obs=("Q", "max"),
            event_start_ts=("timestamp", "min"),
            event_end_ts=("timestamp", "max"),
            n_points=("Q", "size"),
        ).reset_index()

        # duration_days: prefer provided; else compute if allowed
        if "duration_days" in self.events_df.columns:
            dur = self.events_df[["gauge", "event_id", "duration_days"]].copy()
            meta = meta.merge(dur, on=["gauge", "event_id"], how="left")
        elif self.cfg.compute_duration_if_missing and {"event_start", "event_end"}.issubset(subdaily.columns):
            tmp = grp.agg(event_start=("event_start", "first"), event_end=("event_end", "first")).reset_index()
            tmp["duration_days"] = (
                pd.to_datetime(tmp["event_end"]) - pd.to_datetime(tmp["event_start"])
            ).dt.total_seconds() / 86400.0
            meta = meta.merge(tmp[["gauge", "event_id", "duration_days"]], on=["gauge", "event_id"], how="left")
        else:
            meta["duration_days"] = np.nan

        # drop weak/invalid events
        meta = meta[meta["n_points"] >= self.cfg.min_points_per_event].copy()
        meta = meta[np.isfinite(meta["event_peak_obs"]) & (meta["event_peak_obs"] > 0)].copy()
        meta = meta[np.isfinite(meta["duration_days"]) & (meta["duration_days"] > 0)].copy()

        if meta.empty:
            raise RuntimeError("All fetched events were filtered out (too few points / invalid peak / missing duration).")

        # keep only valid events in subdaily
        keep = meta[["gauge", "event_id"]].drop_duplicates()
        subdaily = subdaily.merge(keep, on=["gauge", "event_id"], how="inner")

        # normalize + rel time
        subdaily = subdaily.merge(meta[["gauge", "event_id", "event_peak_obs", "event_start_ts"]], on=["gauge", "event_id"], how="left")
        subdaily["Q_norm"] = subdaily["Q"] / meta.set_index(["gauge","event_id"]).loc[list(zip(subdaily["gauge"], subdaily["event_id"])),"event_peak_obs"].to_numpy()
        # safer version without fancy indexing:
        subdaily["Q_norm"] = subdaily["Q"] / subdaily["event_peak_obs"].clip(lower=self.cfg.eps_peak)

        subdaily["t_hours"] = (subdaily["timestamp"] - subdaily["event_start_ts"]).dt.total_seconds() / 3600.0

        # store
        self.subdaily_raw = subdaily.sort_values(["gauge", "event_id", "timestamp"]).reset_index(drop=True)
        self.event_library = meta.sort_values(["gauge", "duration_days", "event_id"]).reset_index(drop=True)

        # build compact templates per event for fast reuse
        self.templates = self._build_templates(self.subdaily_raw, self.event_library)
        self._templates_sorted = self._prep_sorted_templates(self.templates)

        if self.cfg.verbose:
            n_tpl = sum(len(v) for v in self.templates.values())
            print(f"Built {n_tpl} templates across {len(self.templates)} gauges.")

    def refit_simulations(
        self,
        filtered_df: pd.DataFrame,
        sites: Optional[List[str]] = None,
        id_col: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        For each row in filtered_df and each site:
          - pick closest-duration historic template
          - scale Q_norm by simulated max_intensity
        Returns long df with one row per subdaily timestep per simulated event.
        """
        if not self._templates_sorted:
            raise RuntimeError("Call build_library() before refit_simulations().")

        df = filtered_df.copy()
        if id_col is None:
            # Robust: never depends on index name collisions
            df["sim_row_id"] = np.arange(len(df), dtype=int)
            id_col = "sim_row_id"
        else:
            if id_col not in df.columns:
                raise ValueError(f"id_col='{id_col}' not found in filtered_df.")

        sites = sites or self._infer_sites_from_filtered(df)

        out = []
        for site in sites:
            dur_col = f"{site}_duration_days"
            inten_col = f"{site}_max_intensity"
            freq_col = f"{site}_frequency"

            missing = [c for c in [dur_col, inten_col] if c not in df.columns]
            if missing:
                raise ValueError(f"filtered_df missing required columns for site='{site}': {missing}")

            sdf = df.copy()
            if self.cfg.keep_only_positive_frequency and freq_col in sdf.columns:
                sdf = sdf[sdf[freq_col].fillna(0) > 0]

            sdf = sdf.replace([np.inf, -np.inf], np.nan).dropna(subset=[dur_col, inten_col])
            sdf = sdf[(sdf[dur_col] > 0) & (sdf[inten_col] > 0)]
            if sdf.empty:
                continue

            # match template per row (fast)
            tpl = self._match_templates(
                gauge=site,
                sim_durations=sdf[dur_col].to_numpy(float),
                metric=self.cfg.duration_metric,
            )

            # expand each sim row into its template points (loop is fine; events count << points)
            for sim_row_id, sim_dur, sim_peak, template in zip(
                sdf[id_col].to_numpy(),
                sdf[dur_col].to_numpy(float),
                sdf[inten_col].to_numpy(float),
                tpl,
            ):
                q_sim = template.q_norm * sim_peak

                d = pd.DataFrame({
                    "sim_event_id": f"{site}__{sim_row_id}",
                    id_col: sim_row_id,
                    "gauge": site,
                    "template_event_id": template.event_id,
                    "sim_duration_days": sim_dur,
                    "sim_peak_intensity": sim_peak,
                    "t_hours": template.rel_time_hours,
                    "Q_norm": template.q_norm,
                    "Q_sim": q_sim,
                    "template_duration_days": template.duration_days,
                    "template_peak_obs": template.peak_obs,
                })

                # optional synthetic timestamp axis
                d["timestamp"] = pd.to_datetime(0, unit="s") + pd.to_timedelta(d["t_hours"], unit="h")
                out.append(d)

        if not out:
            return pd.DataFrame(columns=[
                "sim_event_id", id_col, "gauge", "template_event_id",
                "sim_duration_days", "sim_peak_intensity",
                "timestamp", "t_hours", "Q_norm", "Q_sim",
                "template_duration_days", "template_peak_obs",
            ])

        out_df = pd.concat(out, ignore_index=True)
        return out_df.sort_values(["gauge", id_col, "t_hours"]).reset_index(drop=True)

    # ---------------------------------------------------------------------
    # Internals: validation / inference
    # ---------------------------------------------------------------------
    def _validate_events_df(self) -> None:
        req = {"gauge", "start", "end"}
        missing = req - set(self.events_df.columns)
        if missing:
            raise ValueError(f"events_df missing required columns: {missing}")

        # ensure event_id exists
        if "event_id" not in self.events_df.columns:
            s = pd.to_datetime(self.events_df["start"])
            e = pd.to_datetime(self.events_df["end"])
            self.events_df["event_id"] = (
                self.events_df["gauge"].astype(str)
                + "__"
                + s.dt.strftime("%Y%m%d%H%M%S")
                + "_"
                + e.dt.strftime("%Y%m%d%H%M%S")
            )

        # ensure gauges covered
        missing_gauges = sorted(set(self.events_df["gauge"]) - set(self.site_meta.keys()))
        if missing_gauges:
            raise ValueError(f"These gauges are missing from site_meta: {missing_gauges}")

    def _infer_sites_from_filtered(self, df: pd.DataFrame) -> List[str]:
        sites = []
        for c in df.columns:
            if c.endswith("_duration_days"):
                s = c[:-len("_duration_days")]
                if f"{s}_max_intensity" in df.columns:
                    sites.append(s)
        return sorted(set(sites))

    # ---------------------------------------------------------------------
    # Internals: template build & matching
    # ---------------------------------------------------------------------
    def _build_templates(self, subdaily: pd.DataFrame, meta: pd.DataFrame) -> Dict[str, List[HydroTemplate]]:
        meta_idx = meta.set_index(["gauge", "event_id"])
        templates: Dict[str, List[HydroTemplate]] = {}

        for (g, eid), grp in subdaily.groupby(["gauge", "event_id"], sort=False):
            m = meta_idx.loc[(g, eid)]
            rel = grp["t_hours"].to_numpy(float)
            qn = grp["Q_norm"].to_numpy(float)

            # enforce monotone time / drop duplicates if any
            order = np.argsort(rel)
            rel = rel[order]
            qn = qn[order]

            templates.setdefault(g, []).append(
                HydroTemplate(
                    gauge=g,
                    event_id=eid,
                    duration_days=float(m["duration_days"]),
                    peak_obs=float(m["event_peak_obs"]),
                    rel_time_hours=rel,
                    q_norm=qn,
                )
            )
        return templates

    def _prep_sorted_templates(
        self, templates: Dict[str, List[HydroTemplate]]
    ) -> Dict[str, Tuple[np.ndarray, List[HydroTemplate]]]:
        """
        Pre-sort templates by duration for O(N log M) matching.
        Returns gauge -> (durations_sorted, templates_sorted)
        """
        out: Dict[str, Tuple[np.ndarray, List[HydroTemplate]]] = {}
        for g, lst in templates.items():
            lst2 = sorted(lst, key=lambda t: t.duration_days)
            d = np.array([t.duration_days for t in lst2], dtype=float)
            out[g] = (d, lst2)
        return out

    def _match_templates(
        self,
        gauge: str,
        sim_durations: np.ndarray,
        metric: str = "abs",
    ) -> List[HydroTemplate]:
        """
        Fast 1D nearest-neighbor match on duration using searchsorted.
        """
        if gauge not in self._templates_sorted or len(self._templates_sorted[gauge][1]) == 0:
            if not self.cfg.allow_cross_gauge:
                raise RuntimeError(f"No templates available for gauge='{gauge}'.")
            # fallback: pool all templates
            all_tpl = [t for L in self.templates.values() for t in L]
            all_tpl = sorted(all_tpl, key=lambda t: t.duration_days)
            d = np.array([t.duration_days for t in all_tpl], dtype=float)
            durations_sorted, tpl_sorted = d, all_tpl
        else:
            durations_sorted, tpl_sorted = self._templates_sorted[gauge]

        sd = sim_durations.astype(float)

        if metric == "logabs":
            sd = np.log(sd)
            dd = np.log(durations_sorted)
        elif metric == "abs":
            dd = durations_sorted
        else:
            raise ValueError("metric must be 'abs' or 'logabs'.")

        # searchsorted gives insertion positions; compare neighbor(s)
        idx = np.searchsorted(dd, sd, side="left")
        idx0 = np.clip(idx - 1, 0, len(dd) - 1)
        idx1 = np.clip(idx, 0, len(dd) - 1)

        # choose closer of idx0 vs idx1
        choose1 = np.abs(sd - dd[idx1]) < np.abs(sd - dd[idx0])
        best = np.where(choose1, idx1, idx0)

        return [tpl_sorted[i] for i in best]

    # ---------------------------------------------------------------------
    # Internals: your fetch/build logic (kept very close to your version)
    # ---------------------------------------------------------------------
    def _build_subdaily_event_dataset(
        self,
        events_df: pd.DataFrame,
        site_meta: dict,
        parameter_code: str = "00060",
        buffer_hours: int = 0,
        tz: str | None = None,
        verbose: bool = True,
    ) -> pd.DataFrame:
        required = {"gauge", "event_id", "start", "end"}
        missing = required - set(events_df.columns)
        if missing:
            raise ValueError(f"events_df missing required columns: {missing}")

        pieces = []
        n = len(events_df)
        missing_iv = []

        events_df = events_df.reset_index(drop=True)

        for i, row in events_df.iterrows():
            gauge = row["gauge"]
            event_id = row["event_id"]

            site_no = site_meta[gauge]["site"]

            start = pd.to_datetime(row["start"]) - pd.Timedelta(hours=buffer_hours)
            end = pd.to_datetime(row["end"]) + pd.Timedelta(hours=buffer_hours)

            if verbose:
                print(f"[{i+1}/{n}] Fetching IV for {gauge} ({site_no}) {start} → {end} ({event_id})")

            df_iv = self.fetch_iv_window(
                site_no=site_no,
                start=start,
                end=end,
                parameter_code=parameter_code,
                tz=tz,
            )

            if df_iv is None or df_iv.empty:
                if verbose:
                    print(f"  → No IV data found for {gauge} ({site_no}) in window {event_id}")
                missing_iv.append({"gauge": gauge, "event_id": event_id, "start": row["start"], "end": row["end"]})
                continue

            # pick discharge column
            discharge_cols = [
                c for c in df_iv.columns
                if str(c).startswith(parameter_code) and not str(c).endswith("_cd")
            ]
            discharge_cols = [c for c in discharge_cols if np.issubdtype(df_iv[c].dtype, np.number)]

            if discharge_cols:
                qcol = discharge_cols[0]
                out = df_iv[[qcol]].rename(columns={qcol: "Q"})
            else:
                num_cols = df_iv.select_dtypes(include=[np.number]).columns.tolist()
                if not num_cols:
                    raise ValueError(f"No numeric columns returned for gauge={gauge}, event_id={event_id}.")
                out = df_iv[[num_cols[0]]].rename(columns={num_cols[0]: "Q"})

            # datetime -> timestamp
            out = out.reset_index().rename(columns={"datetime": "timestamp"})
            if "timestamp" not in out.columns:
                out = out.rename(columns={out.columns[0]: "timestamp"})

            out["gauge"] = gauge
            out["event_id"] = event_id
            out["event_start"] = pd.to_datetime(row["start"])
            out["event_end"] = pd.to_datetime(row["end"])

            pieces.append(out[["event_id", "gauge", "timestamp", "Q", "event_start", "event_end"]])

        if verbose and missing_iv:
            print("\nSummary: No IV data found for these events:")
            print(pd.DataFrame(missing_iv).to_string(index=False))

        if not pieces:
            return pd.DataFrame(columns=["event_id", "gauge", "timestamp", "Q", "event_start", "event_end"])

        subdaily = pd.concat(pieces, ignore_index=True)
        subdaily["timestamp"] = pd.to_datetime(subdaily["timestamp"])
        subdaily = subdaily.sort_values(["gauge", "event_id", "timestamp"]).reset_index(drop=True)
        return subdaily