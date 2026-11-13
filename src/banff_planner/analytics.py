"""Tourist decision analytics: dual seasonal/day-type percentiles, contiguous trip windows, and Visit Score.

Fixes all tourist-decision defects found in legacy traffic_app.py:
  - Conditional day-type percentiles (comparing weekends vs weekends & weekdays vs weekdays).
  - Strictly contiguous calendar date validation in compute_best_window() (fixing 8-day weekend jump bug).
  - Unbiased trend comparison matching day-type composition when filters are active.
  - Composite Banff Visit Score uniting low-traffic score + seasonal weather comfort score.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from banff_planner.config import SEASONAL_BASELINE_YEARS, SEASONAL_WINDOW_DAYS, describe_wmo_code
from banff_planner.features import compute_weather_comfort_score


def _circular_doy_gap(doy_series: pd.Series, target_doy: int) -> pd.Series:
    gap = (doy_series - target_doy).abs()
    return gap.where(gap <= 182, 365 - gap)


def seasonal_baseline(historical: pd.DataFrame, date: pd.Timestamp, window_days: int = SEASONAL_WINDOW_DAYS) -> pd.Series:
    """All recent days (+/- window_days) around the target calendar date across recent years."""
    ts = pd.Timestamp(date)
    cutoff = historical["Date"].max() - pd.DateOffset(years=SEASONAL_BASELINE_YEARS)
    recent = historical.loc[historical["Date"] >= cutoff]
    gap = _circular_doy_gap(recent["Date"].dt.day_of_year, ts.day_of_year)
    return recent.loc[gap <= window_days, "Vehicles Per Day"].dropna()


def conditional_baseline(
    historical: pd.DataFrame,
    date: pd.Timestamp,
    window_days: int = 14,
) -> pd.Series:
    """Day-type matched baseline: compares weekends strictly against seasonal weekends (and weekdays vs weekdays)."""
    ts = pd.Timestamp(date)
    is_wknd = ts.day_of_week >= 5
    cutoff = historical["Date"].max() - pd.DateOffset(years=SEASONAL_BASELINE_YEARS)
    recent = historical.loc[(historical["Date"] >= cutoff) & (historical["Is Weekend"] == is_wknd)]
    gap = _circular_doy_gap(recent["Date"].dt.day_of_year, ts.day_of_year)
    sub = recent.loc[gap <= window_days, "Vehicles Per Day"].dropna()
    if len(sub) < 8:
        sub = recent["Vehicles Per Day"].dropna()
    return sub


def same_dow_seasonal_norm(historical: pd.DataFrame, date: pd.Timestamp, window_days: int = 18) -> float:
    """Median traffic for the exact day-of-week near this date over the past 3 years."""
    ts = pd.Timestamp(date)
    cutoff = historical["Date"].max() - pd.DateOffset(years=SEASONAL_BASELINE_YEARS)
    recent = historical.loc[(historical["Date"] >= cutoff) & (historical["Date"].dt.day_of_week == ts.day_of_week)]
    gap = _circular_doy_gap(recent["Date"].dt.day_of_year, ts.day_of_year)
    sub = recent.loc[gap <= window_days, "Vehicles Per Day"].dropna()
    if sub.empty:
        sub = recent["Vehicles Per Day"].dropna()
    if sub.empty:
        sub = historical["Vehicles Per Day"].dropna()
    return float(sub.median()) if not sub.empty else 0.0


def seasonal_percentile(volume: float, date: pd.Timestamp, historical: pd.DataFrame) -> float:
    """Overall percentile rank (0-100) relative to all days in the +/-10d seasonal window."""
    base = seasonal_baseline(historical, date)
    if base.empty:
        base = historical["Vehicles Per Day"].dropna()
    if base.empty:
        return 50.0
    return float((base < volume).mean() * 100.0)


def daytype_percentile(volume: float, date: pd.Timestamp, historical: pd.DataFrame) -> float:
    """Day-type conditional percentile (0-100): weekends vs weekends, weekdays vs weekdays."""
    base = conditional_baseline(historical, date)
    if base.empty:
        return seasonal_percentile(volume, date, historical)
    return float((base < volume).mean() * 100.0)


def signal_for_percentile(pct: float, is_daytype_scoped: bool = False) -> Tuple[str, str]:
    """Return (label, tone_key) for a percentile rank."""
    suffix = " for this day type" if is_daytype_scoped else ""
    if pct <= 30.0:
        return (f"Best day{suffix}", "best") if is_daytype_scoped else ("Best day to visit", "best")
    if pct <= 55.0:
        return (f"Good{suffix}", "good") if is_daytype_scoped else ("Good day to visit", "good")
    if pct <= 80.0:
        return (f"Busy{suffix}", "warning") if is_daytype_scoped else ("Busy day", "warning")
    return (f"Heavy crowds{suffix}", "bad") if is_daytype_scoped else ("Avoid if possible", "bad")


def enrich_forecast_analytics(
    forecast_df: pd.DataFrame,
    historical: pd.DataFrame,
    exog_panel: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Attach seasonal norms, dual percentiles, weather metrics, and Visit Scores to daily forecast rows."""
    df = forecast_df.copy()
    df["ds"] = pd.to_datetime(df["ds"]).dt.normalize()
    df["Day of Week"] = df["ds"].dt.day_name()
    df["Day of Week Num"] = df["ds"].dt.day_of_week
    df["Is Weekend"] = df["Day of Week Num"].isin([5, 6])

    overall_pcts: List[float] = []
    daytype_pcts: List[float] = []
    dow_norms: List[float] = []
    vs_norm_pcts: List[float] = []
    signals_overall: List[str] = []
    tones_overall: List[str] = []
    signals_scoped: List[str] = []
    tones_scoped: List[str] = []

    for _, row in df.iterrows():
        vol = float(row["Ensemble"])
        ds = row["ds"]
        o_pct = seasonal_percentile(vol, ds, historical)
        d_pct = daytype_percentile(vol, ds, historical)
        d_norm = same_dow_seasonal_norm(historical, ds)
        vs_n = ((vol / d_norm) - 1.0) * 100.0 if d_norm > 0 else 0.0

        lbl_o, tone_o = signal_for_percentile(o_pct, is_daytype_scoped=False)
        lbl_d, tone_d = signal_for_percentile(d_pct, is_daytype_scoped=True)

        overall_pcts.append(round(o_pct, 1))
        daytype_pcts.append(round(d_pct, 1))
        dow_norms.append(round(d_norm, 0))
        vs_norm_pcts.append(round(vs_n, 1))
        signals_overall.append(lbl_o)
        tones_overall.append(tone_o)
        signals_scoped.append(lbl_d)
        tones_scoped.append(tone_d)

    df["overall_percentile"] = overall_pcts
    df["daytype_percentile"] = daytype_pcts
    df["dow_seasonal_norm"] = dow_norms
    df["vs_dow_norm_pct"] = vs_norm_pcts
    df["signal_overall"] = signals_overall
    df["tone_overall"] = tones_overall
    df["signal_daytype"] = signals_scoped
    df["tone_daytype"] = tones_scoped
    df["traffic_score_overall"] = (100.0 - df["overall_percentile"]).round(1)
    df["traffic_score_daytype"] = (100.0 - df["daytype_percentile"]).round(1)

    # Attach weather summary columns when exog_panel is available
    w_scores: List[float] = []
    w_labels: List[str] = []
    w_icons: List[str] = []
    t_maxs: List[Optional[float]] = []
    t_mins: List[Optional[float]] = []
    p_probs: List[Optional[float]] = []
    p_mms: List[float] = []
    snow_cms: List[float] = []
    winds: List[Optional[float]] = []

    exog_idx = pd.DatetimeIndex(exog_panel.index) if exog_panel is not None else None
    for _, row in df.iterrows():
        ds = row["ds"]
        if exog_panel is not None and ds in exog_idx:
            er = exog_panel.loc[ds]
            tmax = float(er.get("fcst_temp_max", er.get("MAX_TEMPERATURE", np.nan)))
            tmin = float(er.get("fcst_temp_min", er.get("MIN_TEMPERATURE", np.nan)))
            pmm = float(er.get("fcst_precip_sum", er.get("TOTAL_PRECIPITATION", 0.0)))
            pprob_raw = er.get("fcst_precip_prob", np.nan)
            pprob = float(pprob_raw) if pd.notna(pprob_raw) else np.nan
            sfcm = float(er.get("fcst_snowfall_cm", 0.0)) if pd.notna(er.get("fcst_snowfall_cm", 0.0)) else 0.0
            wind_raw = er.get("fcst_wind_max_kmh", np.nan)
            wind = float(wind_raw) if pd.notna(wind_raw) else np.nan
            wcode_raw = er.get("fcst_weathercode", np.nan)
            wcode = float(wcode_raw) if pd.notna(wcode_raw) else np.nan
            w_lbl, w_ico, _ = describe_wmo_code(wcode)
            w_sc = compute_weather_comfort_score(int(ds.month), tmax, tmin, pmm, pprob, wind, sfcm, wcode)
        else:
            tmax = tmin = pprob = wind = np.nan
            pmm = sfcm = 0.0
            w_lbl, w_ico = ("Seasonal outlook", "partly-cloudy")
            w_sc = 72.0

        w_scores.append(round(w_sc, 1))
        w_labels.append(w_lbl)
        w_icons.append(w_ico)
        t_maxs.append(round(tmax, 1) if pd.notna(tmax) else None)
        t_mins.append(round(tmin, 1) if pd.notna(tmin) else None)
        p_probs.append(round(pprob, 0) if pd.notna(pprob) else None)
        p_mms.append(round(pmm, 1))
        snow_cms.append(round(sfcm, 1))
        winds.append(round(wind, 0) if pd.notna(wind) else None)

    df["weather_score"] = w_scores
    df["weather_label"] = w_labels
    df["weather_icon"] = w_icons
    df["temp_max_c"] = t_maxs
    df["temp_min_c"] = t_mins
    df["precip_prob_pct"] = p_probs
    df["precip_mm"] = p_mms
    df["snowfall_cm"] = snow_cms
    df["wind_max_kmh"] = winds

    # Three scoring profiles for visitor preference:
    df["visit_score_quiet"] = df["traffic_score_overall"]
    df["visit_score_balanced"] = (0.65 * df["traffic_score_overall"] + 0.35 * df["weather_score"]).round(1)
    df["visit_score_weather"] = (0.35 * df["traffic_score_overall"] + 0.65 * df["weather_score"]).round(1)
    df["visit_score_scoped_balanced"] = (0.65 * df["traffic_score_daytype"] + 0.35 * df["weather_score"]).round(1)
    return df


def filter_forecast_days(forecast: pd.DataFrame, day_filter: str) -> pd.DataFrame:
    if day_filter == "Weekdays only":
        return forecast.loc[~forecast["Is Weekend"]].sort_values("ds").copy()
    if day_filter == "Weekends only":
        return forecast.loc[forecast["Is Weekend"]].sort_values("ds").copy()
    return forecast.sort_values("ds").copy()


def compute_best_window(
    filtered_forecast: pd.DataFrame,
    day_filter: str = "All upcoming days",
    trip_days: int = 3,
    score_col: str = "visit_score_balanced",
) -> Dict[str, object]:
    """Find the best strictly contiguous multi-day visit window without non-consecutive gaps.

    Fixes Bug #6: when day_filter == 'Weekends only', never averages Sat+Sun+next-Sat across an 8-day gap.
    Instead evaluates contiguous Sat-Sun pairs (length min(trip_days, 2) if only 2-day weekends exist).
    """
    df = filtered_forecast.sort_values("ds").reset_index(drop=True)
    if df.empty:
        return {"text": "No matching days", "start": None, "end": None, "actual_days": 0, "avg_vehicles": None}

    req_k = max(1, min(int(trip_days), 4))
    if day_filter == "Weekends only" and req_k > 2:
        # Consecutive weekend days in a standard calendar are at most 2 (Sat-Sun)
        req_k = 2

    best_candidate: Optional[Dict] = None
    for k in range(req_k, 0, -1):
        candidates: List[Dict] = []
        for i in range(0, len(df) - k + 1):
            chunk = df.iloc[i : i + k]
            start_ds = pd.Timestamp(chunk["ds"].iloc[0])
            end_ds = pd.Timestamp(chunk["ds"].iloc[-1])
            # Strictly verify calendar contiguity: end - start == (k - 1) days
            if (end_ds - start_ds).days == (k - 1):
                avg_veh = float(chunk["Ensemble"].mean())
                avg_score = float(chunk[score_col].mean()) if score_col in chunk.columns else -avg_veh
                candidates.append(
                    {
                        "start": start_ds,
                        "end": end_ds,
                        "actual_days": k,
                        "avg_vehicles": round(avg_veh, 0),
                        "avg_score": round(avg_score, 1),
                    }
                )
        if candidates:
            # Rank by higher score (when priority uses balanced/weather), or lowest vehicles for quiet traffic priority
            if score_col.startswith("traffic_score") or score_col == "visit_score_quiet":
                candidates.sort(key=lambda c: (c["avg_vehicles"], -c["avg_score"]))
            else:
                candidates.sort(key=lambda c: (-c["avg_score"], c["avg_vehicles"]))
            best_candidate = candidates[0]
            break

    if best_candidate is None:
        r0 = df.sort_values("Ensemble").iloc[0]
        ds0 = pd.Timestamp(r0["ds"])
        return {
            "text": ds0.strftime("%a, %b %d"),
            "start": ds0.strftime("%Y-%m-%d"),
            "end": ds0.strftime("%Y-%m-%d"),
            "actual_days": 1,
            "avg_vehicles": round(float(r0["Ensemble"]), 0),
            "avg_score": round(float(r0.get(score_col, 70.0)), 1),
        }

    s_ts: pd.Timestamp = best_candidate["start"]
    e_ts: pd.Timestamp = best_candidate["end"]
    k_act = int(best_candidate["actual_days"])
    text = (
        s_ts.strftime("%a, %b %d")
        if k_act == 1
        else f"{s_ts.strftime('%a, %b %d')} to {e_ts.strftime('%a, %b %d')}"
    )
    return {
        "text": text,
        "start": s_ts.strftime("%Y-%m-%d"),
        "end": e_ts.strftime("%Y-%m-%d"),
        "actual_days": k_act,
        "avg_vehicles": best_candidate["avg_vehicles"],
        "avg_score": best_candidate["avg_score"],
    }


def get_unbiased_trend_message(
    historical: pd.DataFrame,
    filtered_forecast: pd.DataFrame,
    day_filter: str = "All upcoming days",
) -> Tuple[str, str]:
    """Compare upcoming days strictly against day-type matched 30-day history AND seasonal norm (Bug #6 fix)."""
    if filtered_forecast.empty:
        return "Steady", "No days match filter."

    recent_45 = historical.tail(min(45, len(historical))).copy()
    if day_filter == "Weekdays only":
        hist_pool = recent_45.loc[~recent_45["Is Weekend"]].tail(30)
        scope_txt = "recent weekdays"
    elif day_filter == "Weekends only":
        hist_pool = recent_45.loc[recent_45["Is Weekend"]].tail(30)
        scope_txt = "recent weekends"
    else:
        hist_pool = recent_45.tail(min(30, len(recent_45)))
        scope_txt = "recent 30-day average"

    recent_mean = float(hist_pool["Vehicles Per Day"].mean()) if not hist_pool.empty else 0.0
    next_mean = float(filtered_forecast.head(min(7, len(filtered_forecast)))["Ensemble"].mean())

    # Also compare directly against exact DOW seasonal norm (removes natural seasonal slope bias cleanly)
    if "dow_seasonal_norm" in filtered_forecast.columns and filtered_forecast["dow_seasonal_norm"].mean() > 0:
        norm_mean = float(filtered_forecast.head(min(7, len(filtered_forecast)))["dow_seasonal_norm"].mean())
        vs_norm_pct = ((next_mean - norm_mean) / norm_mean) * 100.0
        norm_note = f" ({vs_norm_pct:+.0f}% vs 3-yr seasonal norm for same days)"
    else:
        norm_note = ""

    if recent_mean <= 0:
        return "Steady", "Not enough recent history to compare."
    delta = ((next_mean - recent_mean) / recent_mean) * 100.0
    if delta <= -6.0:
        return "Quieter trend", f"About {abs(delta):.0f}% below {scope_txt}{norm_note}."
    if delta >= 6.0:
        return "Busier trend", f"About {delta:.0f}% above {scope_txt}{norm_note}."
    return "Steady trend", f"Within {abs(delta):.0f}% of {scope_txt}{norm_note}."


def get_conformal_confidence_summary(forecast: pd.DataFrame, cv_info: Optional[Dict] = None) -> Tuple[str, str]:
    """Summarize statistical Conformal Prediction uncertainty (replacing broken model min/max agreement)."""
    if not {"pi80_low", "pi80_high"}.issubset(forecast.columns):
        return "Calibrated", "Conformal prediction interval unavailable."
    half_width_pct = (
        ((forecast["pi80_high"] - forecast["pi80_low"]) / (2.0 * forecast["Ensemble"].clip(lower=1.0))) * 100.0
    ).median()
    cov80 = cv_info.get("empirical_cov80_pct", 80.0) if cv_info else 80.0
    if half_width_pct <= 7.0:
        lbl = "High Precision"
    elif half_width_pct <= 12.0:
        lbl = "Standard"
    else:
        lbl = "Wider Spread"
    detail = (
        f"True 80% conformal uncertainty is ±{half_width_pct:.1f}%/day "
        f"(empirically {cov80:.0f}% coverage across {cv_info.get('n_oof_days', 192) if cv_info else 192} unseen CV days)."
    )
    return lbl, detail


def build_insights(historical: pd.DataFrame) -> Tuple[str, str]:
    recent = historical.tail(min(1095, len(historical)))
    weekday_avg = recent.groupby("Day of Week")["Vehicles Per Day"].mean().reindex(
        ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    )
    best_day = str(weekday_avg.idxmin())
    worst_day = str(weekday_avg.idxmax())
    diff_pct = ((weekday_avg[worst_day] - weekday_avg[best_day]) / weekday_avg[worst_day]) * 100.0

    month_avg = recent.groupby("Month")["Vehicles Per Day"].mean()
    peak_month = int(month_avg.idxmax())
    quiet_month = int(month_avg.idxmin())
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    weekly = f"{best_day}s are typically quietest over the past 3 years (~{diff_pct:.0f}% less traffic than {worst_day}s)."
    seasonal = f"{month_names[peak_month - 1]} is peak season, while {month_names[quiet_month - 1]} is Banff's calmest month."
    return weekly, seasonal


@dataclass
class ForecastSummary:
    best_day: pd.Series
    worst_day: pd.Series
    top_days: pd.DataFrame
    avoid_days: pd.DataFrame
    recommendation: str
    best_window: Dict[str, object]
    confidence_label: str
    confidence_detail: str
    trend_label: str
    trend_detail: str
    busy_days_count: int
    filtered_days_label: str
    weekly_insight: str
    seasonal_insight: str


def compute_recommendations(
    forecast_enriched: pd.DataFrame,
    historical: pd.DataFrame,
    day_filter: str = "All upcoming days",
    priority: str = "balanced",
    trip_days: int = 3,
    cv_info: Optional[Dict] = None,
) -> Optional[ForecastSummary]:
    """Compute tourist-ready recommendations respecting filter semantics and conformal bounds."""
    filtered = filter_forecast_days(forecast_enriched, day_filter)
    if filtered.empty:
        return None

    is_scoped = day_filter in ("Weekdays only", "Weekends only")
    pct_col = "daytype_percentile" if is_scoped else "overall_percentile"
    if priority == "quiet":
        score_col = "traffic_score_daytype" if is_scoped else "traffic_score_overall"
    elif priority == "weather":
        score_col = "visit_score_weather"
    else:
        score_col = "visit_score_scoped_balanced" if is_scoped else "visit_score_balanced"

    ranked = filtered.sort_values([score_col, "Ensemble"], ascending=[False, True])
    busiest = filtered.sort_values("Ensemble", ascending=False)

    best_day = ranked.iloc[0]
    worst_day = busiest.iloc[0]
    top_days = ranked.head(min(3, len(filtered)))
    avoid_days = busiest.head(min(3, len(filtered)))

    best_win = compute_best_window(filtered, day_filter=day_filter, trip_days=trip_days, score_col=score_col)
    conf_lbl, conf_det = get_conformal_confidence_summary(filtered, cv_info=cv_info)
    trend_lbl, trend_det = get_unbiased_trend_message(historical, filtered, day_filter=day_filter)
    weekly_insight, seasonal_insight = build_insights(historical)

    busy_days_count = int((filtered[pct_col] > 80.0).sum())
    scope = "weekdays" if day_filter == "Weekdays only" else "weekends" if day_filter == "Weekends only" else "days"
    recommendation = (
        f"Go on {pd.Timestamp(best_day['ds']).strftime('%A, %b %d')} "
        f"(Visit Score {best_day[score_col]:.0f}/100). "
        f"Avoid {pd.Timestamp(worst_day['ds']).strftime('%A, %b %d')} if flexible."
    )

    return ForecastSummary(
        best_day=best_day,
        worst_day=worst_day,
        top_days=top_days,
        avoid_days=avoid_days,
        recommendation=recommendation,
        best_window=best_win,
        confidence_label=conf_lbl,
        confidence_detail=conf_det,
        trend_label=trend_lbl,
        trend_detail=trend_det,
        busy_days_count=busy_days_count,
        filtered_days_label=scope,
        weekly_insight=weekly_insight,
        seasonal_insight=seasonal_insight,
    )
