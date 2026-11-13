"""Live web exogenous & traffic data connectors with deterministic caching and integrity guards.

Sources integrated programmatically (Zero Waste data is strictly excluded):
  1. Town of Banff Tableau Public (.twb -> .hyper via pantab): West, East, Mountain Ave counters.
  2. Climate: ECCC BANFF CS (3050519) baseline + live ECCC OGC climate-daily +
     Open-Meteo ERA5 Archive (month-bias calibrated gap-fill) + Open-Meteo 16-Day Weather Forecast.
  3. Bank of Canada Valet API: Daily FXUSDCAD + legacy pre-2017 IEXE0101 splice.
  4. Statistics Canada WDS POST API: Alberta CPI (All, Gasoline, Energy, Food), FoodServices,
     Retail, LFS Employment & Unemployment, Average Hourly Wage, US & Overseas Visitors.
  5. Alberta Economic Dashboard API: Highway 1 Banff Park Gates vehicle counts & Park Attendance.
"""

import datetime as dt
import io
import logging
import os
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pantab
import requests
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from banff_planner.config import (
    ALBERTA_HWY_COUNT_URL,
    ALBERTA_PARK_ATTENDANCE_URL,
    BANFF_LAT,
    BANFF_LON,
    BANFF_TABLEAU_WORKBOOK_URL,
    BOC_VALET_FX_URL,
    BOC_VALET_LEGACY_FX_URL,
    CACHE_DIR,
    DEFAULT_HORIZON_DAYS,
    ECCC_CLIMATE_DAILY_URL,
    ECCC_STATION_ID,
    OPEN_METEO_ARCHIVE_URL,
    OPEN_METEO_FORECAST_URL,
    RAW_DATA_DIR,
    RAW_TRAFFIC_CSV,
    RAW_WEATHER_CSV,
    ROOT_TRAFFIC_TSV,
    STATCAN_VECTORS,
    STATCAN_WDS_VECTOR_URL,
)
from banff_planner.data_loader import load_daily_traffic

logger = logging.getLogger(__name__)
REQUEST_HEADERS = {"User-Agent": "BanffVisitPlanner/2.0", "Accept": "application/json"}
TABLEAU_RETRY_STATUSES = {429, 500, 502, 503, 504}


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update(REQUEST_HEADERS)
    return s


def _norm_ts(values) -> pd.DatetimeIndex:
    """Return timezone-naive daily normalized DatetimeIndex with consistent nanosecond unit."""
    dt_idx = pd.DatetimeIndex(pd.to_datetime(values, errors="coerce")).tz_localize(None).normalize()
    try:
        return dt_idx.as_unit("ns")
    except Exception:
        return dt_idx


def _calgary_today() -> pd.Timestamp:
    """Return current normalized local calendar date in America/Edmonton (Mountain Time)."""
    now_mt = pd.Timestamp.now(tz="America/Edmonton").tz_localize(None).normalize()
    try:
        return now_mt.as_unit("ns")
    except Exception:
        return now_mt


# ---------------------------------------------------------------------------
# 1. Tableau Public Traffic Extraction (Counters 1, 2, 3)
# ---------------------------------------------------------------------------
def _fetch_tableau_workbook(
    url: str = BANFF_TABLEAU_WORKBOOK_URL,
    attempts: int = 5,
    backoff_base: float = 5.0,
) -> requests.Response:
    """Download Tableau Public .twb archive retrying rate limits and transient 5xx."""
    sess = _session()
    for attempt in range(1, attempts + 1):
        try:
            response = sess.get(url, headers={"Accept": "*/*"}, timeout=120)
        except requests.RequestException as exc:
            reason = str(exc)
            wait = backoff_base * (2 ** (attempt - 1))
        else:
            if response.status_code not in TABLEAU_RETRY_STATUSES:
                response.raise_for_status()
                return response
            reason = f"HTTP {response.status_code}"
            parsed_wait = 0.0
            retry_after = response.headers.get("Retry-After")
            if retry_after is not None:
                try:
                    parsed_wait = max(0.0, float(retry_after))
                except (TypeError, ValueError):
                    parsed_wait = 0.0
            wait = parsed_wait if parsed_wait > 0.0 else backoff_base * (2 ** (attempt - 1))

        if attempt == attempts:
            raise RuntimeError(f"Tableau fetch failed after {attempts} attempts: {reason}")
        logger.warning("Tableau attempt %d/%d failed (%s); retrying in %.1fs.", attempt, attempts, reason, wait)
        time.sleep(wait)
    raise RuntimeError("Tableau fetch exited retry loop unexpectedly.")


def _extract_daily_traffic_from_hyper_frames(
    frames: Dict[object, pd.DataFrame],
    min_hours_per_day: int = 23,
    start_date: str = "2013-07-01",
) -> pd.DataFrame:
    """Aggregate hourly pantab .hyper tables explicitly by Counter and Date without silent 0-fills."""
    candidate_tables: List[pd.DataFrame] = []
    for _, df in frames.items():
        if {"Counter", "TimeStamp", "TW"}.issubset(df.columns):
            candidate_tables.append(df[["Counter", "TimeStamp", "TW"]].copy())

    if not candidate_tables:
        raise ValueError("No tables containing ['Counter', 'TimeStamp', 'TW'] found in Tableau .hyper extract.")

    raw = pd.concat(candidate_tables, ignore_index=True)
    raw["Counter"] = raw["Counter"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    raw["TimeStamp"] = pd.to_datetime(raw["TimeStamp"], errors="coerce")
    raw["TW"] = pd.to_numeric(raw["TW"], errors="coerce")
    raw = raw.dropna(subset=["TimeStamp", "TW"]).copy()
    raw = raw[raw["Counter"].isin(["1", "2", "3"])].copy()
    if raw.empty:
        raise ValueError("Tableau .hyper extract yielded no rows for counters {'1', '2', '3'}.")

    # Guard against accidental overlapping rows across multiple extract tables
    raw = raw.drop_duplicates(subset=["Counter", "TimeStamp"], keep="first")
    raw["Date"] = raw["TimeStamp"].dt.normalize()
    raw = raw[raw["Date"] >= pd.Timestamp(start_date)].copy()
    if raw.empty:
        raise ValueError(f"Tableau .hyper extract contains no observations on/after {start_date}.")

    agg = (
        raw.groupby(["Counter", "Date"], as_index=False)
        .agg(
            tw_sum=("TW", lambda s: s.sum(min_count=1)),
            n_hours=("TW", "count"),
        )
    )
    piv_sum = agg.pivot(index="Date", columns="Counter", values="tw_sum").sort_index()
    piv_hrs = agg.pivot(index="Date", columns="Counter", values="n_hours").sort_index()

    for cid in ("1", "2", "3"):
        if cid not in piv_sum.columns:
            piv_sum[cid] = np.nan
        if cid not in piv_hrs.columns:
            piv_hrs[cid] = 0

    complete_primary = (
        piv_sum["1"].notna()
        & piv_sum["2"].notna()
        & (piv_hrs["1"] >= min_hours_per_day)
        & (piv_hrs["2"] >= min_hours_per_day)
    )
    valid_dates = piv_sum.index[complete_primary]
    if len(valid_dates) == 0:
        raise ValueError("Tableau .hyper extract contains zero complete dual-counter days.")

    sum_tw = piv_sum.loc[valid_dates, "1"].round().astype("Int64")
    tw_002 = piv_sum.loc[valid_dates, "2"].round().astype("Int64")
    c3 = piv_sum.loc[valid_dates, "3"]
    c3_ok = c3.notna() & (piv_hrs.loc[valid_dates, "3"] >= min_hours_per_day)
    tw_003 = c3.where(c3_ok).round().astype("Int64")
    combined_tw = (tw_002 + sum_tw).astype("Int64")

    denom = (tw_002.astype(float) + sum_tw.astype(float)).replace(0, np.nan)
    ratio = (combined_tw.astype(float) / denom).median()
    if pd.notna(ratio) and ratio > 1.5:
        raise ValueError(f"Tableau extract failed 4x ratio guard (Combined/(West+East) median={ratio:.2f}).")

    idx = pd.DatetimeIndex(valid_dates)
    return pd.DataFrame(
        {
            "Day of Time Stamp": [f"{d.strftime('%B')} {d.day}, {d.year}" for d in idx],
            "Weekday": idx.strftime("%a"),
            "Combined TW": combined_tw.values,
            "Tw (002)": tw_002.values,
            "Tw (003)": tw_003.values,
            "Sum of TW": sum_tw.values,
        }
    )


def fetch_banff_traffic_tableau(
    raw_dir: Path = RAW_DATA_DIR,
    root_tsv: Optional[Path] = ROOT_TRAFFIC_TSV,
    refresh: bool = False,
    url: str = BANFF_TABLEAU_WORKBOOK_URL,
    attempts: int = 5,
    min_hours_per_day: int = 23,
) -> Path:
    """Fetch live Town of Banff traffic counters from Tableau Public (.twb -> .hyper),
    validate rigorously, and atomically update raw CSV + root TW Traffic _data.csv.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / "tw_traffic_data.csv"
    if not refresh:
        if raw_path.exists():
            return raw_path
        if root_tsv is not None and root_tsv.exists():
            return root_tsv

    tmp_hyper_path: Optional[str] = None
    tmp_tsv_path: Optional[Path] = None
    try:
        resp = _fetch_tableau_workbook(url=url, attempts=attempts)
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            hyper_files = [name for name in zf.namelist() if name.endswith(".hyper")]
            if len(hyper_files) != 1:
                raise ValueError(
                    f"Expected exactly one .hyper extract in workbook, found {len(hyper_files)}: {hyper_files}"
                )
            with tempfile.NamedTemporaryFile(delete=False, suffix=".hyper") as tf:
                tf.write(zf.read(hyper_files[0]))
                tmp_hyper_path = tf.name

        frames = pantab.frames_from_hyper(tmp_hyper_path)
        daily_df = _extract_daily_traffic_from_hyper_frames(frames, min_hours_per_day=min_hours_per_day)

        fd, tsv_name = tempfile.mkstemp(dir=raw_dir, prefix=".tw_traffic_", suffix=".tmp")
        os.close(fd)
        tmp_tsv_path = Path(tsv_name)
        daily_df.to_csv(tmp_tsv_path, index=False, sep="\t", encoding="utf-16")

        validated = load_daily_traffic(csv_path=tmp_tsv_path, reindex_daily=False)
        if validated.empty:
            raise ValueError("Validated daily Town traffic series from Tableau extract is empty.")

        os.replace(tmp_tsv_path, raw_path)
        tmp_tsv_path = None
        if root_tsv is not None:
            fd2, root_tmp_name = tempfile.mkstemp(dir=root_tsv.parent, prefix=".tw_root_", suffix=".tmp")
            os.close(fd2)
            root_tmp = Path(root_tmp_name)
            try:
                daily_df.to_csv(root_tmp, index=False, sep="\t", encoding="utf-16")
                os.replace(root_tmp, root_tsv)
            except Exception:
                root_tmp.unlink(missing_ok=True)
                raise

        logger.info(
            "Updated %s from Tableau Public (%d complete days through %s).",
            raw_path.name,
            len(daily_df),
            validated["Date"].max().strftime("%Y-%m-%d"),
        )
        return raw_path
    except Exception as exc:
        if raw_path.exists():
            logger.warning("Tableau live fetch failed (%s); falling back to %s.", exc, raw_path)
            return raw_path
        if root_tsv is not None and root_tsv.exists():
            logger.warning("Tableau live fetch failed (%s); falling back to %s.", exc, root_tsv)
            return root_tsv
        raise RuntimeError(f"Tableau fetch failed and no fallback CSV exists: {exc}") from exc
    finally:
        if tmp_hyper_path and os.path.exists(tmp_hyper_path):
            os.remove(tmp_hyper_path)
        if tmp_tsv_path and tmp_tsv_path.exists():
            tmp_tsv_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# 2. Macro & Tourism APIs (StatCan WDS, BoC Valet FX, Alberta Dashboard)
# ---------------------------------------------------------------------------
def fetch_statcan_vectors(
    latest_n: int = 180,
    cache_dir: Path = CACHE_DIR,
    refresh: bool = False,
) -> pd.DataFrame:
    """Fetch verified monthly StatCan macro/CPI/labour/tourism series indexed by vectorId."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "statcan_macro.csv"
    if cache_path.exists() and not refresh:
        df = pd.read_csv(cache_path)
        df["ds"] = _norm_ts(df["ds"])
        return df.set_index("ds").sort_index()

    sess = _session()
    payload = [{"vectorId": int(vid), "latestN": int(latest_n)} for vid in STATCAN_VECTORS]
    try:
        resp = sess.post(STATCAN_WDS_VECTOR_URL, json=payload, timeout=30)
        resp.raise_for_status()
        items = resp.json()
    except Exception as exc:
        if cache_path.exists():
            logger.warning("StatCan live fetch failed (%s); falling back to cache.", exc)
            df = pd.read_csv(cache_path)
            df["ds"] = _norm_ts(df["ds"])
            return df.set_index("ds").sort_index()
        raise RuntimeError(f"StatCan fetch failed and no cache exists: {exc}") from exc

    series_frames: List[pd.DataFrame] = []
    for item in items:
        if item.get("status") != "SUCCESS":
            continue
        obj = item.get("object", {})
        vid = int(obj["vectorId"])
        col_name = STATCAN_VECTORS.get(vid)
        if not col_name:
            continue
        pts = obj.get("vectorDataPoint", [])
        rows = []
        for p in pts:
            val = p.get("value")
            ref = p.get("refPer")
            if ref and val is not None:
                ds_ts = (pd.to_datetime(ref) + pd.offsets.MonthEnd(0)).tz_localize(None).normalize()
                rows.append({"ds": ds_ts, col_name: float(val)})
        if rows:
            sdf = pd.DataFrame(rows).drop_duplicates("ds").set_index("ds")
            series_frames.append(sdf)

    if not series_frames:
        if cache_path.exists():
            df = pd.read_csv(cache_path)
            df["ds"] = _norm_ts(df["ds"])
            return df.set_index("ds").sort_index()
        raise RuntimeError("StatCan API returned zero valid series.")

    macro = pd.concat(series_frames, axis=1).sort_index()
    macro.index = _norm_ts(macro.index)
    macro = macro.asfreq("ME")

    if {"AB_FoodServices_Receipts_SA", "AB_CPI_All"}.issubset(macro.columns):
        macro["AB_FoodSvc_Real"] = macro["AB_FoodServices_Receipts_SA"] / macro["AB_CPI_All"] * 100.0
    if {"AB_Retail_Total_SA", "AB_CPI_All"}.issubset(macro.columns):
        macro["AB_Retail_Real"] = macro["AB_Retail_Total_SA"] / macro["AB_CPI_All"] * 100.0
    if {"AB_AvgHourlyWage", "AB_CPI_All"}.issubset(macro.columns):
        macro["AB_Real_Wage"] = macro["AB_AvgHourlyWage"] / macro["AB_CPI_All"] * 100.0
    if {"AB_CPI_Gasoline", "AB_CPI_All"}.issubset(macro.columns):
        macro["AB_Gas_Real"] = macro["AB_CPI_Gasoline"] / macro["AB_CPI_All"] * 100.0
    if {"AB_Visitors_US", "AB_Visitors_Overseas"}.issubset(macro.columns):
        macro["AB_Visitors_Intl"] = macro["AB_Visitors_US"] + macro["AB_Visitors_Overseas"]

    macro.index.name = "ds"
    macro.to_csv(cache_path)
    return macro


def fetch_boc_fx(
    start_date: str = "2013-01-01",
    cache_dir: Path = CACHE_DIR,
    refresh: bool = False,
) -> pd.DataFrame:
    """Fetch Bank of Canada FXUSDCAD (+ legacy pre-2017 IEXE0101 splice) daily & cache daily + monthly."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    daily_cache = cache_dir / "boc_fx_daily.csv"
    monthly_cache = cache_dir / "boc_fx.csv"

    if not refresh and daily_cache.exists():
        df = pd.read_csv(daily_cache)
        df["ds"] = _norm_ts(df["ds"])
        return df.set_index("ds").sort_index()

    if not refresh and monthly_cache.exists():
        # Clean fallback if only monthly FX seed exists
        df_m = pd.read_csv(monthly_cache)
        df_m["ds"] = _norm_ts(df_m["ds"])
        df_m = df_m.set_index("ds").sort_index()
        return df_m

    sess = _session()
    try:
        resp = sess.get(f"{BOC_VALET_FX_URL}?start_date={start_date}", timeout=25)
        resp.raise_for_status()
        obs = resp.json().get("observations", [])
    except Exception as exc:
        if daily_cache.exists():
            logger.warning("BoC Valet fetch failed (%s); falling back to daily cache.", exc)
            df = pd.read_csv(daily_cache)
            df["ds"] = _norm_ts(df["ds"])
            return df.set_index("ds").sort_index()
        if monthly_cache.exists():
            logger.warning("BoC Valet fetch failed (%s); falling back to monthly cache.", exc)
            df_m = pd.read_csv(monthly_cache)
            df_m["ds"] = _norm_ts(df_m["ds"])
            return df_m.set_index("ds").sort_index()
        raise RuntimeError(f"BoC Valet fetch failed and no cache exists: {exc}") from exc

    rows = []
    for o in obs:
        v = o.get("FXUSDCAD", {})
        val = v.get("v") if isinstance(v, dict) else v
        if o.get("d") and val is not None:
            rows.append({"ds": pd.to_datetime(o["d"]).tz_localize(None).normalize(), "FXUSDCAD": float(val)})
    fx_d = pd.DataFrame(rows).drop_duplicates("ds").set_index("ds").sort_index()

    # Splice pre-2017 BoC legacy noon USD/CAD (IEXE0101) cleanly by overlap ratio
    if fx_d.index.min() > pd.Timestamp("2016-12-31"):
        try:
            r_leg = sess.get(
                f"{BOC_VALET_LEGACY_FX_URL}?start_date={start_date}&end_date=2017-04-30",
                timeout=20,
            )
            if r_leg.ok:
                leg_obs = r_leg.json().get("observations", [])
                l_rows = []
                for o in leg_obs:
                    v = o.get("IEXE0101", {})
                    val = v.get("v") if isinstance(v, dict) else v
                    if o.get("d") and val is not None:
                        l_rows.append({"ds": pd.to_datetime(o["d"]).tz_localize(None).normalize(), "FX_LEG": float(val)})
                if l_rows:
                    ldf = pd.DataFrame(l_rows).drop_duplicates("ds").set_index("ds").sort_index()
                    overlap = fx_d.join(ldf, how="inner").dropna()
                    scale = float((overlap["FXUSDCAD"] / overlap["FX_LEG"]).median()) if not overlap.empty else 1.0
                    pre = (ldf.loc[ldf.index < fx_d.index.min(), "FX_LEG"] * scale).rename("FXUSDCAD").to_frame()
                    fx_d = pd.concat([pre, fx_d]).sort_index()
        except Exception as exc:
            logger.info("BoC legacy pre-2017 FX splice skipped (%s).", exc)

    fx_d.index = _norm_ts(fx_d.index)
    fx_d.index.name = "ds"
    fx_d.to_csv(daily_cache)

    # Also keep monthly cache updated
    tmp = fx_d.copy()
    tmp["ym"] = tmp.index.to_period("M").to_timestamp("M")
    counts = tmp.groupby("ym")["FXUSDCAD"].count()
    fx_m = tmp.groupby("ym")[["FXUSDCAD"]].mean()
    fx_m = fx_m[counts >= 5].sort_index()
    fx_m.index.name = "ds"
    fx_m.to_csv(monthly_cache)
    return fx_d


def fetch_alberta_tourism(
    cache_dir: Path = CACHE_DIR,
    refresh: bool = False,
) -> pd.DataFrame:
    """Fetch Banff National Park Attendance and Hwy 1 Banff Park Gates counts from Alberta Dashboard."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "alberta_tourism.csv"
    if cache_path.exists() and not refresh:
        df = pd.read_csv(cache_path)
        df["ds"] = _norm_ts(df["ds"])
        return df.set_index("ds").sort_index()

    sess = _session()
    try:
        r_park = sess.get(ALBERTA_PARK_ATTENDANCE_URL, timeout=25)
        r_park.raise_for_status()
        park_df = pd.DataFrame(r_park.json())
        park_df["ds"] = (pd.to_datetime(park_df["Date"]) + pd.offsets.MonthEnd(0)).dt.tz_localize(None).dt.normalize()
        park_m = (
            park_df.groupby("ds", as_index=True)["Value"]
            .mean()
            .rename("Banff_Park_Attendance")
            .to_frame()
        )

        r_hwy = sess.get(ALBERTA_HWY_COUNT_URL, timeout=25)
        r_hwy.raise_for_status()
        hwy_df = pd.DataFrame(r_hwy.json())
        banff_hwy = hwy_df[
            (hwy_df["TourismName"] == "Hwy 1 - Banff Park Gates")
            & (hwy_df["Type"] == "Total Vehicles")
        ].copy()
        banff_hwy["ds"] = (pd.to_datetime(banff_hwy["Date"]) + pd.offsets.MonthEnd(0)).dt.tz_localize(None).dt.normalize()
        hwy_m = (
            banff_hwy.groupby("ds", as_index=True)["Value"]
            .mean()
            .rename("Hwy1_Banff_Vehicles")
            .to_frame()
        )

        merged = park_m.join(hwy_m, how="outer").sort_index()
        merged.index = _norm_ts(merged.index)
        merged = merged.asfreq("ME")
    except Exception as exc:
        if cache_path.exists():
            logger.warning("Alberta dashboard fetch failed (%s); falling back to cache.", exc)
            df = pd.read_csv(cache_path)
            df["ds"] = _norm_ts(df["ds"])
            return df.set_index("ds").sort_index()
        raise RuntimeError(f"Alberta dashboard fetch failed and no cache exists: {exc}") from exc

    merged.index.name = "ds"
    merged.to_csv(cache_path)
    return merged


# ---------------------------------------------------------------------------
# 3. Daily Weather Builder: BANFF CS + ECCC OGC + Open-Meteo ERA5 + 16d NWP
# ---------------------------------------------------------------------------
def load_historical_weather(csv_path: Optional[Path] = None) -> pd.DataFrame:
    """Load baseline local daily weather (BANFF CS 3050519)."""
    path = Path(csv_path) if csv_path else RAW_WEATHER_CSV
    if not path.exists():
        return pd.DataFrame(
            columns=[
                "LOCAL_DATE",
                "MEAN_TEMPERATURE",
                "MAX_TEMPERATURE",
                "MIN_TEMPERATURE",
                "TOTAL_PRECIPITATION",
                "SNOW_ON_GROUND",
            ]
        )
    w = pd.read_csv(path)
    w["LOCAL_DATE"] = _norm_ts(w["LOCAL_DATE"])
    keep = [
        "LOCAL_DATE",
        "MEAN_TEMPERATURE",
        "MAX_TEMPERATURE",
        "MIN_TEMPERATURE",
        "TOTAL_PRECIPITATION",
        "SNOW_ON_GROUND",
    ]
    w = w[[c for c in keep if c in w.columns]].drop_duplicates("LOCAL_DATE").sort_values("LOCAL_DATE")
    return w.reset_index(drop=True)


def _fetch_eccc_daily_updates(start_date: str = "2025-01-06") -> pd.DataFrame:
    """Fetch recent daily climate records for BANFF CS (3050519) from ECCC OGC API."""
    sess = _session()
    end_date = (_calgary_today() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    all_rows = []
    offset = 0
    limit = 500
    while True:
        params = {
            "CLIMATE_IDENTIFIER": ECCC_STATION_ID,
            "datetime": f"{start_date}/{end_date}",
            "limit": limit,
            "offset": offset,
            "f": "json",
        }
        resp = sess.get(ECCC_CLIMATE_DAILY_URL, params=params, timeout=25)
        resp.raise_for_status()
        feats = resp.json().get("features", [])
        if not feats:
            break
        for feat in feats:
            props = feat.get("properties", {})
            all_rows.append(
                {
                    "LOCAL_DATE": props.get("LOCAL_DATE"),
                    "MEAN_TEMPERATURE": props.get("MEAN_TEMPERATURE"),
                    "MAX_TEMPERATURE": props.get("MAX_TEMPERATURE"),
                    "MIN_TEMPERATURE": props.get("MIN_TEMPERATURE"),
                    "TOTAL_PRECIPITATION": props.get("TOTAL_PRECIPITATION"),
                    "SNOW_ON_GROUND": props.get("SNOW_ON_GROUND"),
                }
            )
        if len(feats) < limit:
            break
        offset += limit

    if not all_rows:
        return pd.DataFrame()
    df = pd.DataFrame(all_rows)
    df["LOCAL_DATE"] = _norm_ts(df["LOCAL_DATE"])
    return df


def _fetch_openmeteo_archive(start_date: str = "2013-07-01", end_date: Optional[str] = None) -> pd.DataFrame:
    """Fetch daily ERA5 reanalysis weather for Banff coordinates from Open-Meteo Archive."""
    if end_date is None:
        end_date = (_calgary_today() - pd.Timedelta(days=5)).strftime("%Y-%m-%d")
    sess = _session()
    params = {
        "latitude": round(BANFF_LAT, 4),
        "longitude": round(BANFF_LON, 4),
        "start_date": start_date,
        "end_date": end_date,
        "daily": "weathercode,temperature_2m_mean,temperature_2m_max,temperature_2m_min,precipitation_sum,snowfall_sum,windspeed_10m_max,sunshine_duration",
        "timezone": "America/Edmonton",
    }
    resp = sess.get(OPEN_METEO_ARCHIVE_URL, params=params, timeout=30)
    resp.raise_for_status()
    daily = resp.json().get("daily", {})
    df = pd.DataFrame(daily)
    df["LOCAL_DATE"] = _norm_ts(df["time"])
    return df.drop(columns=["time"])


def _fetch_openmeteo_forecast(past_days: int = 10, forecast_days: int = 16) -> pd.DataFrame:
    """Fetch Open-Meteo 16-Day Weather Forecast + past_days bridge for Town of Banff."""
    sess = _session()
    params = {
        "latitude": round(BANFF_LAT, 4),
        "longitude": round(BANFF_LON, 4),
        "past_days": past_days,
        "forecast_days": forecast_days,
        "daily": (
            "weathercode,temperature_2m_max,temperature_2m_min,temperature_2m_mean,"
            "precipitation_sum,precipitation_probability_max,snowfall_sum,"
            "windspeed_10m_max,sunshine_duration"
        ),
        "timezone": "America/Edmonton",
    }
    resp = sess.get(OPEN_METEO_FORECAST_URL, params=params, timeout=25)
    resp.raise_for_status()
    daily = resp.json().get("daily", {})
    df = pd.DataFrame(daily)
    df["LOCAL_DATE"] = _norm_ts(df["time"])
    return df.drop(columns=["time"])


def fetch_weather_daily(
    start_date: str = "2013-07-01",
    horizon_days: int = DEFAULT_HORIZON_DAYS,
    cache_dir: Path = CACHE_DIR,
    refresh: bool = False,
) -> pd.DataFrame:
    """Build gap-free daily Banff weather from start_date through today + horizon_days.

    Combines:
      - ECCC BANFF CS (3050519) historical + live OGC daily updates
      - Month-bias-adjusted Open-Meteo ERA5 historical gap-fill
      - Open-Meteo 16-day forecast (weathercode, precip probability, wind, temp min/max/mean, snow, sunshine)
      - Proper summer missing SNOW_ON_GROUND = 0.0 fix + 10-year circular daily climatology
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "weather_daily.csv"
    today_mt = _calgary_today()
    min_required_end = today_mt + pd.Timedelta(days=horizon_days)

    if cache_path.exists() and not refresh:
        cached = pd.read_csv(cache_path)
        cached["ds"] = _norm_ts(cached["ds"])
        cached = cached.set_index("ds").sort_index()
        if cached.index.max() >= min_required_end - pd.Timedelta(days=1):
            return cached

    hist = load_historical_weather()
    if not hist.empty:
        hist = hist[hist["LOCAL_DATE"] >= pd.Timestamp(start_date)].copy()
        last_hist_str = (hist["LOCAL_DATE"].max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    else:
        last_hist_str = start_date

    try:
        eccc_new = _fetch_eccc_daily_updates(start_date=last_hist_str)
    except Exception as exc:
        logger.warning("ECCC OGC daily fetch failed (%s); continuing with local weather.", exc)
        eccc_new = pd.DataFrame()

    daily = pd.concat([hist, eccc_new], ignore_index=True)
    if not daily.empty:
        daily = daily.drop_duplicates("LOCAL_DATE").sort_values("LOCAL_DATE").reset_index(drop=True)
    else:
        daily = pd.DataFrame(columns=["LOCAL_DATE"])

    om_fcst = pd.DataFrame()
    try:
        om_fcst = _fetch_openmeteo_forecast(past_days=14, forecast_days=max(16, horizon_days))
    except Exception as exc:
        logger.warning("Open-Meteo 16-day forecast fetch failed (%s).", exc)

    end_target = min_required_end
    if not om_fcst.empty:
        end_target = max(end_target, om_fcst["LOCAL_DATE"].max())

    full_days = _norm_ts(pd.date_range(start=start_date, end=end_target, freq="D"))
    daily = daily.set_index("LOCAL_DATE") if not daily.empty else pd.DataFrame(index=pd.DatetimeIndex([]))
    daily.index = _norm_ts(daily.index)
    daily = daily.reindex(full_days)
    daily.index.name = "ds"

    # Merge Open-Meteo ERA5 archive + Open-Meteo forecast tail
    om_era5 = pd.DataFrame()
    try:
        era5_end = (today_mt - pd.Timedelta(days=5)).strftime("%Y-%m-%d")
        om_era5 = _fetch_openmeteo_archive(start_date=start_date, end_date=era5_end)
    except Exception as exc:
        logger.warning("Open-Meteo ERA5 archive fetch failed (%s).", exc)

    om_frames = []
    if not om_era5.empty:
        om_era5["om_is_forecast"] = False
        om_frames.append(om_era5)
    if not om_fcst.empty:
        om_fcst["om_is_forecast"] = om_fcst["LOCAL_DATE"] >= today_mt
        om_frames.append(om_fcst)

    if om_frames:
        om_all = (
            pd.concat(om_frames, ignore_index=True)
            .drop_duplicates("LOCAL_DATE", keep="last")
            .set_index("LOCAL_DATE")
            .sort_index()
        )
        om_all.index = _norm_ts(om_all.index)
        daily = daily.join(om_all, how="left")
    else:
        # Cached fallback if offline cleanly
        if cache_path.exists():
            logger.warning("Both ECCC & Open-Meteo failed; using cached daily weather.")
            cached = pd.read_csv(cache_path)
            cached["ds"] = _norm_ts(cached["ds"])
            return cached.set_index("ds").sort_index()

    # Estimate station-vs-ERA5 monthly elevation bias across overlapping valid ECCC days
    mo_bias_mean = pd.Series(dtype=float)
    mo_bias_max = pd.Series(dtype=float)
    mo_bias_min = pd.Series(dtype=float)
    if {"MEAN_TEMPERATURE", "temperature_2m_mean"}.issubset(daily.columns):
        overlap = daily.dropna(subset=["MEAN_TEMPERATURE", "temperature_2m_mean"]).copy()
        if not overlap.empty:
            overlap["mo"] = overlap.index.month
            mo_bias_mean = overlap.groupby("mo").apply(
                lambda g: float((g["MEAN_TEMPERATURE"] - g["temperature_2m_mean"]).mean()),
                include_groups=False,
            )
        overlap_hi = daily.dropna(subset=["MAX_TEMPERATURE", "temperature_2m_max"]).copy()
        if not overlap_hi.empty:
            overlap_hi["mo"] = overlap_hi.index.month
            mo_bias_max = overlap_hi.groupby("mo").apply(
                lambda g: float((g["MAX_TEMPERATURE"] - g["temperature_2m_max"]).mean()),
                include_groups=False,
            )
        overlap_lo = daily.dropna(subset=["MIN_TEMPERATURE", "temperature_2m_min"]).copy()
        if not overlap_lo.empty:
            overlap_lo["mo"] = overlap_lo.index.month
            mo_bias_min = overlap_lo.groupby("mo").apply(
                lambda g: float((g["MIN_TEMPERATURE"] - g["temperature_2m_min"]).mean()),
                include_groups=False,
            )

    mo_series = pd.Series(daily.index.month, index=daily.index)
    for col, om_col, bias_map in [
        ("MEAN_TEMPERATURE", "temperature_2m_mean", mo_bias_mean),
        ("MAX_TEMPERATURE", "temperature_2m_max", mo_bias_max),
        ("MIN_TEMPERATURE", "temperature_2m_min", mo_bias_min),
    ]:
        if col not in daily.columns:
            daily[col] = np.nan
        if om_col in daily.columns:
            miss = daily[col].isna() & daily[om_col].notna()
            if miss.any():
                bias = mo_series.loc[miss].map(bias_map).fillna(1.1) if not bias_map.empty else 1.1
                daily.loc[miss, col] = daily.loc[miss, om_col] + bias

    if "TOTAL_PRECIPITATION" not in daily.columns:
        daily["TOTAL_PRECIPITATION"] = np.nan
    if "precipitation_sum" in daily.columns:
        miss_p = daily["TOTAL_PRECIPITATION"].isna() & daily["precipitation_sum"].notna()
        if miss_p.any():
            daily.loc[miss_p, "TOTAL_PRECIPITATION"] = daily.loc[miss_p, "precipitation_sum"]

    # Preserve Town-level raw Open-Meteo display fields cleanly
    daily["fcst_weathercode"] = daily["weathercode"] if "weathercode" in daily.columns else np.nan
    daily["fcst_temp_max"] = daily.get("temperature_2m_max", daily["MAX_TEMPERATURE"]).combine_first(daily["MAX_TEMPERATURE"])
    daily["fcst_temp_min"] = daily.get("temperature_2m_min", daily["MIN_TEMPERATURE"]).combine_first(daily["MIN_TEMPERATURE"])
    daily["fcst_temp_mean"] = daily.get("temperature_2m_mean", daily["MEAN_TEMPERATURE"]).combine_first(daily["MEAN_TEMPERATURE"])
    daily["fcst_precip_sum"] = daily.get("precipitation_sum", daily["TOTAL_PRECIPITATION"]).combine_first(daily["TOTAL_PRECIPITATION"])
    daily["fcst_precip_prob"] = daily["precipitation_probability_max"] if "precipitation_probability_max" in daily.columns else np.nan
    daily["fcst_snowfall_cm"] = daily["snowfall_sum"] if "snowfall_sum" in daily.columns else 0.0
    daily["fcst_wind_max_kmh"] = daily["windspeed_10m_max"] if "windspeed_10m_max" in daily.columns else np.nan
    if "sunshine_duration" in daily.columns:
        daily["fcst_sunshine_hours"] = daily["sunshine_duration"] / 3600.0
    else:
        daily["fcst_sunshine_hours"] = np.nan

    # Fill isolated station gaps via circular 10-year daily normal if anything beyond forecast end is needed
    clim_base = daily.loc["2016-01-01":"2025-12-31"].copy() if "2016-01-01" <= str(daily.index.max().date()) else daily.copy()
    if clim_base["MEAN_TEMPERATURE"].notna().any():
        clim_base["doy"] = np.clip(clim_base.index.dayofyear, 1, 365)
        doy_norm_raw = clim_base.groupby("doy")[["MEAN_TEMPERATURE", "MAX_TEMPERATURE", "MIN_TEMPERATURE", "TOTAL_PRECIPITATION"]].mean().reindex(range(1, 366))
        # Circular 15-day smooth climatology
        padded = pd.concat([doy_norm_raw.iloc[-7:], doy_norm_raw, doy_norm_raw.iloc[:7]])
        doy_smooth = padded.rolling(15, center=True, min_periods=1).mean().iloc[7:-7]
        doy_smooth.index = range(1, 366)
    else:
        doy_smooth = pd.DataFrame(
            {"MEAN_TEMPERATURE": 3.0, "MAX_TEMPERATURE": 9.0, "MIN_TEMPERATURE": -3.0, "TOTAL_PRECIPITATION": 1.2},
            index=range(1, 366),
        )

    doy_idx = pd.Series(np.clip(daily.index.dayofyear, 1, 365), index=daily.index)
    for col in ["MEAN_TEMPERATURE", "MAX_TEMPERATURE", "MIN_TEMPERATURE", "TOTAL_PRECIPITATION"]:
        daily[col] = daily[col].interpolate(method="time", limit=14)
        still_na = daily[col].isna()
        if still_na.any():
            daily.loc[still_na, col] = doy_idx.loc[still_na].map(doy_smooth[col]).values

    daily["TEMP_NORMAL_10Y"] = doy_idx.map(doy_smooth["MEAN_TEMPERATURE"]).astype(float)
    daily["TEMP_ANOM"] = daily["MEAN_TEMPERATURE"] - daily["TEMP_NORMAL_10Y"]
    daily["PRECIP_NORMAL_10Y"] = doy_idx.map(doy_smooth["TOTAL_PRECIPITATION"]).astype(float)
    daily["TOTAL_PRECIPITATION"] = daily["TOTAL_PRECIPITATION"].clip(lower=0.0).fillna(0.0)
    daily["fcst_snowfall_cm"] = daily["fcst_snowfall_cm"].clip(lower=0.0).fillna(0.0)
    daily["fcst_temp_max"] = daily["fcst_temp_max"].combine_first(daily["MAX_TEMPERATURE"])
    daily["fcst_temp_min"] = daily["fcst_temp_min"].combine_first(daily["MIN_TEMPERATURE"])
    daily["fcst_temp_mean"] = daily["fcst_temp_mean"].combine_first(daily["MEAN_TEMPERATURE"])
    daily["fcst_precip_sum"] = daily["fcst_precip_sum"].combine_first(daily["TOTAL_PRECIPITATION"])

    # Proper Summer Snow-on-Ground Fix: warm months or MEAN_TEMPERATURE > 2.5C -> 0.0 (never winter global mean)
    if "SNOW_ON_GROUND" not in daily.columns:
        daily["SNOW_ON_GROUND"] = np.nan
    warm_mask = daily["SNOW_ON_GROUND"].isna() & (
        mo_series.isin([5, 6, 7, 8, 9]) | (daily["MEAN_TEMPERATURE"] > 2.5)
    )
    daily.loc[warm_mask, "SNOW_ON_GROUND"] = 0.0
    daily["SNOW_ON_GROUND"] = daily["SNOW_ON_GROUND"].interpolate(method="time", limit=7)

    # Physical snow depth rollout for trailing/future days missing ECCC snow depth
    if daily["SNOW_ON_GROUND"].isna().any():
        snow_vals = daily["SNOW_ON_GROUND"].to_numpy(dtype=float, copy=True)
        t_vals = daily["MEAN_TEMPERATURE"].to_numpy(dtype=float)
        sf_vals = daily["fcst_snowfall_cm"].to_numpy(dtype=float)
        mo_vals = daily.index.month.to_numpy()
        for i in range(len(snow_vals)):
            if np.isnan(snow_vals[i]):
                if mo_vals[i] in (5, 6, 7, 8, 9) or t_vals[i] > 2.5:
                    snow_vals[i] = 0.0
                else:
                    prev = snow_vals[i - 1] if i > 0 and not np.isnan(snow_vals[i - 1]) else 0.0
                    melt = max(0.0, t_vals[i] - 0.5) * 0.65
                    snow_vals[i] = max(0.0, prev - melt + sf_vals[i] * 0.6)
        daily["SNOW_ON_GROUND"] = snow_vals

    daily["is_realized_weather"] = daily.index < today_mt
    daily["is_forecast_weather"] = daily.index >= today_mt
    if "om_is_forecast" in daily.columns:
        daily["has_live_weather_forecast"] = daily["om_is_forecast"].fillna(False) & daily["fcst_weathercode"].notna()
    else:
        daily["has_live_weather_forecast"] = False

    # Derive weather comfort features useful for ML & tourist UX
    daily["is_freezing"] = (daily["MAX_TEMPERATURE"] <= 0.0).astype(float)
    daily["is_heavy_precip"] = (daily["TOTAL_PRECIPITATION"] >= 5.0).astype(float)
    daily["is_snow_day"] = (daily["fcst_snowfall_cm"] >= 1.0).astype(float)
    daily["diurnal_range"] = (daily["MAX_TEMPERATURE"] - daily["MIN_TEMPERATURE"]).clip(lower=0.0)

    keep_cols = [
        "MEAN_TEMPERATURE",
        "MAX_TEMPERATURE",
        "MIN_TEMPERATURE",
        "TOTAL_PRECIPITATION",
        "SNOW_ON_GROUND",
        "TEMP_NORMAL_10Y",
        "TEMP_ANOM",
        "PRECIP_NORMAL_10Y",
        "diurnal_range",
        "is_freezing",
        "is_heavy_precip",
        "is_snow_day",
        "fcst_weathercode",
        "fcst_temp_max",
        "fcst_temp_min",
        "fcst_temp_mean",
        "fcst_precip_sum",
        "fcst_precip_prob",
        "fcst_snowfall_cm",
        "fcst_wind_max_kmh",
        "fcst_sunshine_hours",
        "is_realized_weather",
        "is_forecast_weather",
        "has_live_weather_forecast",
    ]
    out = daily[[c for c in keep_cols if c in daily.columns]].copy()
    out.index.name = "ds"
    out.to_csv(cache_path)
    return out


# ---------------------------------------------------------------------------
# 4. Daily Exogenous Panel Merger (Climate + Daily FX + Macro + Alberta Tourism)
# ---------------------------------------------------------------------------
def _project_monthly_ets(s: pd.Series, target_end: pd.Timestamp, seasonal: bool = False) -> pd.Series:
    """Safely extrapolate a monthly macro/tourism indicator with damped ETS to target_end."""
    clean = s.dropna().copy()
    clean.index = _norm_ts(clean.index)
    clean = clean.sort_index().asfreq("ME")
    if clean.empty:
        return clean
    if clean.index.max() >= target_end:
        return clean.loc[:target_end]
    h = len(pd.date_range(clean.index.max() + pd.offsets.MonthEnd(1), target_end, freq="ME"))
    if h <= 0:
        return clean
    scale = float(clean.abs().median())
    scale = scale if scale > 1e-6 else 1.0
    scaled = clean / scale
    try:
        if seasonal and len(scaled) >= 36:
            fit = ExponentialSmoothing(
                scaled,
                trend="add",
                damped_trend=True,
                seasonal="mul" if (scaled > 0).all() else "add",
                seasonal_periods=12,
                initialization_method="heuristic",
            ).fit(optimized=True)
        elif len(scaled) >= 18:
            fit = ExponentialSmoothing(
                scaled, trend="add", damped_trend=True, initialization_method="heuristic"
            ).fit(optimized=True)
        else:
            last_val = float(clean.iloc[-1])
            fut_idx = _norm_ts(pd.date_range(clean.index.max() + pd.offsets.MonthEnd(1), periods=h, freq="ME"))
            return pd.concat([clean, pd.Series(last_val, index=fut_idx, name=clean.name)])
        fc = fit.forecast(h) * scale
        fc.index = _norm_ts(pd.date_range(clean.index.max() + pd.offsets.MonthEnd(1), periods=h, freq="ME"))
        if (clean > 0).all():
            fc = fc.clip(lower=clean.min() * 0.5)
        return pd.concat([clean, fc])
    except Exception:
        last_val = float(clean.iloc[-1])
        fut_idx = _norm_ts(pd.date_range(clean.index.max() + pd.offsets.MonthEnd(1), periods=h, freq="ME"))
        return pd.concat([clean, pd.Series(last_val, index=fut_idx, name=clean.name)])


def build_daily_exogenous_panel(
    start_date: str = "2013-07-01",
    horizon_days: int = DEFAULT_HORIZON_DAYS,
    cache_dir: Path = CACHE_DIR,
    refresh: bool = False,
) -> pd.DataFrame:
    """Build unified daily exogenous panel from start_date through today + horizon_days.

    Aligns:
      - Daily Banff climate & 16-day Open-Meteo NWP forecast
      - Daily BoC USD/CAD exchange rate (+ forward-filled business days onto weekends, zero lookahead)
      - Monthly StatCan indicators & Alberta Hwy 1 / Park Attendance smoothly extended onto daily grid
      - Explicit provenance flags (`is_realized_weather`, `is_realized_fx`, `is_realized_macro`,
        `is_realized_tourism`, `is_realized_hwy1`)
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    panel_cache = cache_dir / "daily_exogenous_panel.csv"
    today_mt = _calgary_today()
    min_required_end = today_mt + pd.Timedelta(days=horizon_days)

    if panel_cache.exists() and not refresh:
        cached = pd.read_csv(panel_cache)
        cached["ds"] = _norm_ts(cached["ds"])
        cached = cached.set_index("ds").sort_index()
        if cached.index.max() >= min_required_end - pd.Timedelta(days=1):
            for c in [c for c in cached.columns if c.startswith("is_") or c.startswith("has_")]:
                cached[c] = cached[c].astype(bool)
            return cached

    weather_d = fetch_weather_daily(
        start_date=start_date,
        horizon_days=horizon_days,
        cache_dir=cache_dir,
        refresh=refresh,
    )
    fx_raw = fetch_boc_fx(start_date="2013-01-01", cache_dir=cache_dir, refresh=refresh)
    macro_m = fetch_statcan_vectors(latest_n=180, cache_dir=cache_dir, refresh=refresh)
    tourism_m = fetch_alberta_tourism(cache_dir=cache_dir, refresh=refresh)

    end_ts = max(weather_d.index.max(), min_required_end)
    full_daily_idx = _norm_ts(pd.date_range(start=start_date, end=end_ts, freq="D"))
    out = weather_d.reindex(full_daily_idx).copy()

    # 1. Align Daily Bank of Canada FX without lookahead (ffill business days; carry latest 5d mean forward)
    if not fx_raw.empty and "FXUSDCAD" in fx_raw.columns:
        fx_obs = fx_raw["FXUSDCAD"].dropna().sort_index()
        fx_obs.index = _norm_ts(fx_obs.index)
        last_fx_obs = fx_obs.index.max()
        # Allow daily alignment covering 180d before start_date if present
        fx_full_idx = _norm_ts(pd.date_range(min(pd.Timestamp(start_date), fx_obs.index.min()), end_ts, freq="D"))
        fx_daily = fx_obs.reindex(fx_full_idx).ffill()
        # Hold constant recent 5-day average beyond last observed quote
        recent_fx = float(fx_obs.iloc[-5:].mean()) if len(fx_obs) >= 5 else float(fx_obs.iloc[-1])
        fx_daily.loc[fx_daily.index > last_fx_obs] = recent_fx
        fx_daily = fx_daily.bfill()
        out["FXUSDCAD"] = fx_daily.reindex(full_daily_idx)
        out["is_realized_fx"] = full_daily_idx <= last_fx_obs
        # Smooth lagged FX averages matching tourist booking lead (30d & 90d lagged by 60d)
        fx_90d_ma = fx_daily.rolling(90, min_periods=1).mean()
        out["fx_90d_lag60"] = fx_90d_ma.shift(60).reindex(full_daily_idx).bfill()
        out["fx_30d_lag14"] = fx_daily.rolling(30, min_periods=1).mean().shift(14).reindex(full_daily_idx).bfill()
    else:
        out["FXUSDCAD"] = 1.38
        out["fx_90d_lag60"] = 1.38
        out["fx_30d_lag14"] = 1.38
        out["is_realized_fx"] = False

    # 2. Align Monthly StatCan & Alberta Tourism indicators onto daily grid smoothly
    target_m_end = (end_ts + pd.offsets.MonthEnd(2)).tz_localize(None).normalize()
    monthly_start = (pd.Timestamp(start_date) - pd.offsets.MonthEnd(2)).tz_localize(None).normalize()
    full_m_idx = _norm_ts(pd.date_range(monthly_start, target_m_end, freq="ME"))

    macro_cols = [
        c
        for c in [
            "AB_CPI_All",
            "AB_CPI_Gasoline",
            "AB_Gas_Real",
            "AB_FoodSvc_Real",
            "AB_Retail_Real",
            "AB_LFS_UnempRate_SA",
            "AB_AvgHourlyWage",
            "AB_Real_Wage",
            "AB_Visitors_US",
            "AB_Visitors_Overseas",
            "AB_Visitors_Intl",
            "CA_GDP_AccomFood_M",
        ]
        if c in macro_m.columns
    ]
    seasonal_macro = {"AB_Visitors_US", "AB_Visitors_Overseas", "AB_Visitors_Intl"}

    last_macro_cutoff = macro_m["AB_FoodServices_Receipts_SA"].dropna().index.max() if "AB_FoodServices_Receipts_SA" in macro_m else macro_m.index.max()
    last_park_cutoff = tourism_m["Banff_Park_Attendance"].dropna().index.max() if "Banff_Park_Attendance" in tourism_m else tourism_m.index.max()
    last_hwy_cutoff = tourism_m["Hwy1_Banff_Vehicles"].dropna().index.max() if "Hwy1_Banff_Vehicles" in tourism_m else tourism_m.index.max()

    monthly_proj = pd.DataFrame(index=full_m_idx)
    for col in macro_cols:
        s_src = macro_m[col].dropna()
        if not s_src.empty:
            monthly_proj[col] = _project_monthly_ets(s_src, target_m_end, seasonal=(col in seasonal_macro)).reindex(full_m_idx).bfill()

    for col in ["Banff_Park_Attendance", "Hwy1_Banff_Vehicles"]:
        if col in tourism_m.columns:
            s_src = tourism_m[col].dropna()
            s_fit = s_src.loc["2021-07-01":] if len(s_src.loc["2021-07-01":]) >= 36 else s_src.replace(0.0, np.nan).interpolate()
            ext = _project_monthly_ets(s_fit, target_m_end, seasonal=True).reindex(full_m_idx)
            comb = s_src.reindex(full_m_idx)
            comb.loc[comb.index > s_src.index.max()] = ext.loc[comb.index > s_src.index.max()]
            monthly_proj[col] = comb.bfill()

    # Interpolate monthly series smoothly to midpoint-anchored daily curve, keeping end-of-month exactness
    # plus compute derived lagged macro features (shifted >=45 days so published macro is causal!)
    monthly_mid = monthly_proj.copy()
    monthly_mid.index = _norm_ts(monthly_mid.index - pd.Timedelta(days=14))
    daily_macro_grid = _norm_ts(pd.date_range(monthly_mid.index.min(), max(end_ts, monthly_mid.index.max()), freq="D"))
    daily_macro = monthly_mid.reindex(daily_macro_grid).interpolate(method="time").ffill().bfill()

    for col in monthly_proj.columns:
        out[col] = daily_macro[col].reindex(full_daily_idx).values

    # Recompute derived real macro series smoothly on full daily range
    if {"AB_CPI_Gasoline", "AB_CPI_All"}.issubset(out.columns):
        out["AB_Gas_Real"] = out["AB_CPI_Gasoline"] / out["AB_CPI_All"] * 100.0

    # Lagged realized macro indicators (causal: avoids dependence on un-released same-month macro)
    if "AB_Gas_Real" in daily_macro.columns:
        out["AB_Gas_Real_lag45"] = daily_macro["AB_Gas_Real"].shift(45).reindex(full_daily_idx).bfill()
    if "AB_LFS_UnempRate_SA" in daily_macro.columns:
        out["AB_Unemp_lag45"] = daily_macro["AB_LFS_UnempRate_SA"].shift(45).reindex(full_daily_idx).bfill()
    if "AB_FoodSvc_Real" in daily_macro.columns:
        out["AB_FoodSvc_Real_lag60"] = daily_macro["AB_FoodSvc_Real"].shift(60).reindex(full_daily_idx).bfill()
    if "Hwy1_Banff_Vehicles" in daily_macro.columns:
        # Express Hwy 1 relative to same day prior year (regime momentum without seasonal duplication)
        hwy_s = daily_macro["Hwy1_Banff_Vehicles"]
        hwy_yoy = (hwy_s / hwy_s.shift(365).replace(0, np.nan)).fillna(1.0)
        out["Hwy1_YoY_lag45"] = hwy_yoy.shift(45).reindex(full_daily_idx).fillna(1.0)
    if "Banff_Park_Attendance" in daily_macro.columns:
        pk_s = daily_macro["Banff_Park_Attendance"]
        pk_yoy = (pk_s / pk_s.shift(365).replace(0, np.nan)).fillna(1.0)
        out["ParkAtt_YoY_lag45"] = pk_yoy.shift(45).reindex(full_daily_idx).fillna(1.0)
    if "AB_Visitors_Intl" in daily_macro.columns:
        vi_s = daily_macro["AB_Visitors_Intl"]
        out["AB_Visitors_Intl_lag75"] = vi_s.shift(75).reindex(full_daily_idx).bfill()

    out["is_realized_macro"] = full_daily_idx <= _norm_ts([last_macro_cutoff])[0]
    out["is_realized_tourism"] = full_daily_idx <= _norm_ts([last_park_cutoff])[0]
    out["is_realized_hwy1"] = full_daily_idx <= _norm_ts([last_hwy_cutoff])[0]

    # Guard against any stray NaNs on continuous numeric exogenous fields (leaving fcst_weathercode/prob intact for past)
    numeric_model_cols = [
        "MEAN_TEMPERATURE",
        "MAX_TEMPERATURE",
        "MIN_TEMPERATURE",
        "TOTAL_PRECIPITATION",
        "SNOW_ON_GROUND",
        "TEMP_NORMAL_10Y",
        "TEMP_ANOM",
        "FXUSDCAD",
        "fx_90d_lag60",
        "fx_30d_lag14",
    ]
    for c in numeric_model_cols:
        if c in out.columns:
            out[c] = out[c].ffill().bfill()

    out.index.name = "ds"
    out.to_csv(panel_cache)
    return out


def summarize_data_freshness(
    traffic_df: pd.DataFrame,
    exog_panel: pd.DataFrame,
    cache_dir: Path = CACHE_DIR,
) -> Dict[str, object]:
    """Inspect realized observation cutoffs across all live programmatic feeds."""
    obs_traffic = traffic_df[traffic_df.get("is_observed", True)] if "is_observed" in traffic_df.columns else traffic_df
    last_traffic = pd.Timestamp(obs_traffic["Date"].max()).strftime("%Y-%m-%d") if not obs_traffic.empty else "n/a"
    last_mtn_ave = (
        pd.Timestamp(obs_traffic.loc[obs_traffic["Mountain_Ave"].notna(), "Date"].max()).strftime("%Y-%m-%d")
        if "Mountain_Ave" in obs_traffic.columns and obs_traffic["Mountain_Ave"].notna().any()
        else "n/a"
    )

    def _max_true(flag_col: str) -> str:
        if flag_col not in exog_panel.columns:
            return "n/a"
        sub = exog_panel.index[exog_panel[flag_col].astype(bool)]
        return pd.Timestamp(sub.max()).strftime("%Y-%m-%d") if len(sub) > 0 else "n/a"

    weather_end = (
        pd.Timestamp(exog_panel.index[exog_panel["has_live_weather_forecast"].astype(bool)].max()).strftime("%Y-%m-%d")
        if "has_live_weather_forecast" in exog_panel.columns
        and exog_panel["has_live_weather_forecast"].astype(bool).any()
        else pd.Timestamp(exog_panel.index.max()).strftime("%Y-%m-%d")
    )

    latest_fx = (
        round(float(exog_panel.loc[exog_panel["is_realized_fx"].astype(bool), "FXUSDCAD"].iloc[-1]), 4)
        if "is_realized_fx" in exog_panel.columns
        and exog_panel["is_realized_fx"].astype(bool).any()
        and "FXUSDCAD" in exog_panel.columns
        else None
    )

    last_statcan_real = _max_true("is_realized_macro")
    last_alberta_att = _max_true("is_realized_tourism")
    last_alberta_hwy = _max_true("is_realized_hwy1")

    # Also read latest exact un-interpolated monthly macro indicators if cache exists
    latest_macro_vals: Dict[str, Optional[float]] = {}
    statcan_cache = cache_dir / "statcan_macro.csv"
    if statcan_cache.exists():
        sm = pd.read_csv(statcan_cache)
        for col in ("AB_LFS_UnempRate_SA", "AB_Gas_Real", "AB_CPI_Gasoline", "AB_Visitors_Intl"):
            if col in sm.columns and sm[col].notna().any():
                latest_macro_vals[col] = round(float(sm[col].dropna().iloc[-1]), 2)

    return {
        "generated_at_utc": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M UTC"),
        "generated_date_mt": _calgary_today().strftime("%Y-%m-%d"),
        "traffic_days": int(len(obs_traffic)),
        "traffic_start_date": pd.Timestamp(obs_traffic["Date"].min()).strftime("%Y-%m-%d") if not obs_traffic.empty else "n/a",
        "traffic_last_observed": last_traffic,
        "mountain_ave_last_observed": last_mtn_ave,
        "weather_realized_through": _max_true("is_realized_weather"),
        "weather_forecast_through": weather_end,
        "fx_last_observed": _max_true("is_realized_fx"),
        "fx_latest_usdcad": latest_fx,
        "statcan_macro_through": last_statcan_real,
        "alberta_hwy1_through": last_alberta_hwy,
        "alberta_park_attendance_through": last_alberta_att,
        "latest_macro_snapshot": latest_macro_vals,
        "zero_waste_excluded": True,
    }

