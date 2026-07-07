# check_gauges.py

from __future__ import annotations

import re
from io import StringIO
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple, Union, Iterable, List

import numpy as np
import pandas as pd
import requests


@dataclass
class MississippiGaugeScanResult:
    count: int
    meta: Dict[str, Dict[str, str]]
    df: pd.DataFrame
    debug_df: Optional[pd.DataFrame] = None


class MississippiGaugeChecker:
    """
    Pull + filter NWIS DV discharge gauges in (approx) Mississippi basin HUC2s.

    Core:
      - scan() -> MississippiGaugeScanResult (meta dict + filtered POR df)
      - filter_for_overlap() -> further restrict to compatible gauges
      - fetch_site_coords() -> coords df (robust chunking)
      - plot() -> optional (geopandas + geodatasets)
    """

    def __init__(
        self,
        huc2_list: Sequence[str] = ("05", "06", "07", "08", "10", "11"),
        timeout: int = 120,
        base_url: str = "https://waterservices.usgs.gov/nwis/site/",
    ):
        self.huc2_list = tuple(huc2_list)
        self.timeout = int(timeout)
        self.base_url = base_url.rstrip("/") + "/"

        self.last_result: Optional[MississippiGaugeScanResult] = None
        self.last_coords: Optional[pd.DataFrame] = None

    # ----------------------------
    # Internal helpers
    # ----------------------------
    @staticmethod
    def _slugify_station_name(station_nm: str) -> str:
        s = station_nm.lower()
        s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
        s = re.sub(r"_near_|_at_|_nr_|_blw_|_above_|_below_", "_", s)
        return s[:60]

    @staticmethod
    def _rdb_to_df_keep_all_tables(text: str) -> pd.DataFrame:
        lines = [ln for ln in text.splitlines() if not ln.startswith("#") and ln.strip() != ""]
        if not lines:
            return pd.DataFrame()
        return pd.read_csv(StringIO("\n".join(lines)), sep="\t")

    @staticmethod
    def _rdb_por_block_fallback(text: str) -> pd.DataFrame:
        raw = [ln for ln in text.splitlines() if not ln.startswith("#")]
        header_idxs = [
            i for i, ln in enumerate(raw)
            if ("begin_date" in ln and "end_date" in ln and "count_nu" in ln and "site_no" in ln)
        ]
        if not header_idxs:
            return pd.DataFrame()
        start_i = header_idxs[-1]
        block = "\n".join([ln for ln in raw[start_i:] if ln.strip() != ""])
        return pd.read_csv(StringIO(block), sep="\t")

    def _get_huc2_por_table(self, huc2: str) -> pd.DataFrame:
        params = {
            "format": "rdb",
            "huc": huc2,
            "siteType": "ST",
            "hasDataTypeCd": "dv",
            "seriesCatalogOutput": "true",
        }
        r = requests.get(self.base_url, params=params, timeout=self.timeout)
        r.raise_for_status()

        df = self._rdb_to_df_keep_all_tables(r.text)

        needed = {"site_no", "station_nm", "data_type_cd", "parm_cd", "stat_cd",
                  "begin_date", "end_date", "count_nu"}
        if not needed.issubset(df.columns):
            df2 = self._rdb_por_block_fallback(r.text)
            if not df2.empty:
                df = df2

        return df

    @staticmethod
    def _make_meta_from_por(por: pd.DataFrame) -> Dict[str, Dict[str, str]]:
        meta: Dict[str, Dict[str, str]] = {}
        for _, row in por.iterrows():
            station_nm = str(row.get("station_nm", row["site_no"]))
            key = MississippiGaugeChecker._slugify_station_name(station_nm)
            if key in meta:
                key = f"{key}_{row['site_no']}"
            meta[key] = {
                "site": str(row["site_no"]),
                "start": pd.Timestamp(row["begin_date"]).strftime("%Y-%m-%d"),
                "end": pd.Timestamp(row["end_date"]).strftime("%Y-%m-%d"),
            }
        return meta

    # ----------------------------
    # Public API
    # ----------------------------
    def scan(
        self,
        min_years: int = 90,
        completeness: float = 0.995,
        *,
        # harden for loader compatibility
        min_end: Optional[str] = None,  # e.g. "2020-01-01" to avoid 2004/2008-ending gauges
        site_no_regex: str = r"^\d+$",  # exclude '15s' etc
        allowed_site_no_lengths: Sequence[int] = (8,),  # default to 8-digit streamgage ids
        must_include_sites: Sequence[str] = ("07010000", "05420500", "06893000", "05288500"),
        return_debug: bool = True,
        parm_cd: str = "00060",
        stat_cd: str = "00003",
        data_type_cd: str = "dv",
    ) -> MississippiGaugeScanResult:
        """
        Pull POR tables by HUC2 and filter down to "good" DV discharge gauges.

        Added guards:
          - only numeric site_no matching regex
          - only allowed lengths (default 8-digit, which avoids many odd IDs)
          - optional min_end cutoff
        """
        dfs = []
        for huc2 in self.huc2_list:
            df = self._get_huc2_por_table(huc2)
            if df is None or df.empty:
                continue
            df["huc2"] = huc2
            dfs.append(df)

        if not dfs:
            res = MississippiGaugeScanResult(count=0, meta={}, df=pd.DataFrame(),
                                             debug_df=pd.DataFrame() if return_debug else None)
            self.last_result = res
            return res

        por = pd.concat(dfs, ignore_index=True)

        # normalize
        por["site_no"] = por["site_no"].astype(str)
        por["data_type_cd"] = por["data_type_cd"].astype(str).str.lower()
        por["parm_cd"] = por["parm_cd"].astype(str)
        por["stat_cd"] = por["stat_cd"].astype(str)

        por["begin_date"] = pd.to_datetime(por["begin_date"], errors="coerce")
        por["end_date"] = pd.to_datetime(por["end_date"], errors="coerce")
        por["count_nu"] = pd.to_numeric(por["count_nu"], errors="coerce")
        por = por.dropna(subset=["begin_date", "end_date", "count_nu"])

        # debug snapshot before the tight filters (so you can inspect missing majors)
        debug_df = None
        if return_debug and must_include_sites:
            must = set(str(s) for s in must_include_sites)
            debug_df = por[por["site_no"].isin(must)].copy()

        # filter to numeric IDs (exclude non-numeric site IDs)
        rx = re.compile(site_no_regex)
        por = por[por["site_no"].map(lambda s: bool(rx.match(s)))].copy()

        # filter to allowed lengths (default 8-digit streamgage ids)
        allowed_lens = set(int(x) for x in allowed_site_no_lengths)
        por = por[por["site_no"].str.len().isin(allowed_lens)].copy()

        # keep daily discharge mean (dv, 00060, 00003)
        por = por[
            (por["data_type_cd"] == str(data_type_cd).lower()) &
            (por["parm_cd"] == str(parm_cd)) &
            (por["stat_cd"] == str(stat_cd))
        ].copy()

        # duration + continuity
        por["expected_days"] = (por["end_date"] - por["begin_date"]).dt.days + 1
        por = por[por["expected_days"] > 0].copy()
        por["years"] = por["expected_days"] / 365.2425
        por["completeness"] = por["count_nu"] / por["expected_days"]

        # choose one row per site (best coverage)
        por = (
            por.sort_values(["site_no", "count_nu", "expected_days", "completeness"],
                            ascending=[True, False, False, False])
              .drop_duplicates(subset=["site_no"], keep="first")
        )

        # optional: ensure ends are “recent enough”
        if min_end is not None:
            min_end_ts = pd.Timestamp(min_end)
            por = por[por["end_date"] >= min_end_ts].copy()

        # thresholds
        por = por[
            (por["years"] >= float(min_years)) &
            (por["completeness"] >= float(completeness))
        ].copy()

        meta = self._make_meta_from_por(por)
        res = MississippiGaugeScanResult(
            count=len(meta),
            meta=meta,
            df=por.reset_index(drop=True),
            debug_df=debug_df.reset_index(drop=True) if isinstance(debug_df, pd.DataFrame) else debug_df,
        )
        self.last_result = res
        return res

    def filter_for_overlap(
        self,
        result: MississippiGaugeScanResult,
        *,
        climate_start: Union[str, pd.Timestamp],
        climate_end: Union[str, pd.Timestamp],
        min_overlap_years: float = 30.0,
        enforce_common_overlap: bool = True,
    ) -> MississippiGaugeScanResult:
        """
        Ensure what we return will load cleanly in HydroClimateDataset.merge_and_clip().

        Uses POR begin/end (cheap, no DV download):
          - overlap with climate window
          - optional common-overlap across all gauges
          - minimum overlap length (years)
        """
        por = result.df.copy()
        if por.empty:
            return result

        climate_start = pd.Timestamp(climate_start)
        climate_end = pd.Timestamp(climate_end)

        # overlap with climate for each gauge
        por["overlap_start"] = por["begin_date"].clip(lower=climate_start)
        por["overlap_end"] = por["end_date"].clip(upper=climate_end)
        por["overlap_days"] = (por["overlap_end"] - por["overlap_start"]).dt.days + 1
        por = por[por["overlap_days"] > 0].copy()
        por["overlap_years"] = por["overlap_days"] / 365.2425

        por = por[por["overlap_years"] >= float(min_overlap_years)].copy()

        if enforce_common_overlap and not por.empty:
            common_start = por["overlap_start"].max()
            common_end = por["overlap_end"].min()
            common_days = (common_end - common_start).days + 1
            if common_days <= 0:
                # return empty but with debug info
                out = MississippiGaugeScanResult(count=0, meta={}, df=por.reset_index(drop=True),
                                                 debug_df=result.debug_df)
                self.last_result = out
                return out

            # keep only gauges that fully cover the common interval
            por = por[(por["overlap_start"] <= common_start) & (por["overlap_end"] >= common_end)].copy()

        # rebuild meta
        meta = self._make_meta_from_por(por)
        out = MississippiGaugeScanResult(
            count=len(meta),
            meta=meta,
            df=por.reset_index(drop=True),
            debug_df=result.debug_df,
        )
        self.last_result = out
        return out

    def fetch_site_coords(
        self,
        site_nos: Union[Sequence[str], pd.Series],
        chunk_size: int = 200,
    ) -> pd.DataFrame:
        """
        Robust coords fetch:
          - prefilters to numeric-only site ids
          - auto-splits chunks on HTTP 400/404 to avoid hard failure
        """
        site_nos = [str(s) for s in list(site_nos)]
        site_nos = [s for s in site_nos if re.fullmatch(r"\d+", s)]  # keep numeric only
        if not site_nos:
            return pd.DataFrame(columns=["site_no", "station_nm", "dec_lat_va", "dec_long_va", "state_cd", "huc_cd"])

        def _fetch_chunk(chunk: List[str]) -> pd.DataFrame:
            params = {"format": "rdb", "sites": ",".join(chunk), "siteOutput": "expanded"}
            r = requests.get(self.base_url, params=params, timeout=self.timeout)
            r.raise_for_status()
            txt = "\n".join(ln for ln in r.text.splitlines() if not ln.startswith("#") and ln.strip())
            df = pd.read_csv(StringIO(txt), sep="\t")
            keep = [c for c in ["site_no", "station_nm", "dec_lat_va", "dec_long_va", "state_cd", "huc_cd"] if c in df.columns]
            return df[keep].copy()

        out = []
        stack = [site_nos[i:i + int(chunk_size)] for i in range(0, len(site_nos), int(chunk_size))]

        while stack:
            chunk = stack.pop()
            if not chunk:
                continue
            try:
                out.append(_fetch_chunk(chunk))
            except requests.HTTPError as e:
                # If chunk is too big or contains one bad ID, split it
                if len(chunk) == 1:
                    # skip the single offending id
                    continue
                mid = len(chunk) // 2
                stack.append(chunk[:mid])
                stack.append(chunk[mid:])

        if not out:
            return pd.DataFrame(columns=["site_no", "station_nm", "dec_lat_va", "dec_long_va", "state_cd", "huc_cd"])

        coords = pd.concat(out, ignore_index=True).drop_duplicates(subset=["site_no"])
        coords["dec_lat_va"] = pd.to_numeric(coords["dec_lat_va"], errors="coerce")
        coords["dec_long_va"] = pd.to_numeric(coords["dec_long_va"], errors="coerce")
        coords = coords.dropna(subset=["dec_lat_va", "dec_long_va"]).copy()
        coords["site_no"] = coords["site_no"].astype(str)

        self.last_coords = coords.reset_index(drop=True)
        return self.last_coords

    def plot(
        self,
        coords: Optional[pd.DataFrame] = None,
        title: str = "Gauge locations",
        figsize: Tuple[int, int] = (12, 7),
        xlim: Tuple[float, float] = (-125, -66),
        ylim: Tuple[float, float] = (24, 50),
        point_size: float = 10.0,
    ):
        if coords is None:
            if self.last_coords is not None:
                coords = self.last_coords
            elif self.last_result is not None and not self.last_result.df.empty:
                coords = self.fetch_site_coords(self.last_result.df["site_no"].astype(str).unique().tolist())
            else:
                raise ValueError("No coords provided and no previous scan/coords available.")

        import matplotlib.pyplot as plt
        import geopandas as gpd
        from shapely.geometry import Point
        import geodatasets

        gdf = gpd.GeoDataFrame(
            coords.copy(),
            geometry=[Point(xy) for xy in zip(coords["dec_long_va"], coords["dec_lat_va"])],
            crs="EPSG:4326",
        )

        land = gpd.read_file(geodatasets.get_path("naturalearth.land")).to_crs("EPSG:4326")

        fig, ax = plt.subplots(figsize=figsize)
        land.plot(ax=ax, linewidth=0.2, edgecolor="black", facecolor="none")
        ax.scatter(gdf.geometry.x, gdf.geometry.y, s=point_size)

        ax.set_title(title)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        plt.show()
        return ax
