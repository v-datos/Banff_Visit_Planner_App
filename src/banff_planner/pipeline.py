"""End-to-end daily forecasting orchestrator, conformal calibration, deliverables export & site builder."""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from banff_planner.analytics import (
    build_insights,
    enrich_forecast_analytics,
    get_conformal_confidence_summary,
)
from banff_planner.config import (
    CACHE_DIR,
    DEFAULT_HORIZON_DAYS,
    OUTPUTS_DIR,
    ROOT_PREDICTIONS_CSV,
    SEASONAL_BASELINE_YEARS,
    SITE_DIR,
)
from banff_planner.data_loader import load_daily_traffic
from banff_planner.exogenous import (
    _calgary_today,
    build_daily_exogenous_panel,
    fetch_banff_traffic_tableau,
    summarize_data_freshness,
)
from banff_planner.models import (
    BASE_MODEL_NAMES,
    CANDIDATE_ENSEMBLE_MODELS,
    compute_daily_driver_attributions,
    fit_predict_daily_models,
    predict_entrance_breakdown,
)
from banff_planner.evaluation import compute_conformal_intervals, run_expanding_cv
from banff_planner.site_builder import build_static_site

logger = logging.getLogger(__name__)


def _json_safe(obj: Any) -> Any:
    """Recursively convert numpy/pandas scalars & replace non-finite floats with None for valid JSON."""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        val = float(obj)
        return None if (math.isnan(val) or math.isinf(val)) else val
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (pd.Timestamp,)):
        return obj.strftime("%Y-%m-%d")
    if obj is None:
        return None
    try:
        if pd.isna(obj):
            return None
    except Exception:
        pass
    return obj


def _season_tag_for_md(month: int, day: int) -> str:
    if (month == 12 and day >= 23) or (month == 1 and day <= 3):
        return "Christmas & New Year Peak"
    if month in (7, 8):
        return "Peak Summer"
    if (month == 9 and day >= 15) or (month == 10 and day <= 10):
        return "Larch Season Peak"
    if month in (5, 6) or (month == 9 and day < 15):
        return "Shoulder & Hiking"
    if month == 10 and day > 10 or month == 11:
        return "Quiet Autumn Shoulder"
    return "Winter & Ski Season"


def build_annual_planner_data(
    traffic_df: pd.DataFrame,
    exog_panel: pd.DataFrame,
) -> Dict[str, Any]:
    """Compute trailing 3-year 365-day seasonal profile, DOW averages, and monthly climatology."""
    obs = traffic_df[traffic_df.get("is_observed", True)].copy() if "is_observed" in traffic_df.columns else traffic_df.copy()
    obs["Date"] = pd.to_datetime(obs["Date"]).dt.normalize()
    cutoff = obs["Date"].max() - pd.DateOffset(years=SEASONAL_BASELINE_YEARS)
    recent = obs[obs["Date"] >= cutoff].copy()
    if len(recent) < 180:
        recent = obs.copy()

    # Join daily climate normals
    ex = exog_panel.copy()
    ex.index = pd.to_datetime(ex.index).tz_localize(None).normalize()
    recent = recent.merge(
        ex[["MAX_TEMPERATURE", "MIN_TEMPERATURE", "TOTAL_PRECIPITATION"]],
        left_on="Date",
        right_index=True,
        how="left",
    )

    recent["month"] = recent["Date"].dt.month
    recent["day"] = recent["Date"].dt.day
    # Fold Feb 29 into Feb 28 for clean non-leap 365-day calendar lookup
    feb29 = (recent["month"] == 2) & (recent["day"] == 29)
    recent.loc[feb29, "day"] = 28

    ref_dates = pd.date_range("2025-01-01", "2025-12-31", freq="D")
    daily_agg = (
        recent.groupby(["month", "day"], as_index=False)
        .agg(
            avg_veh=("Vehicles Per Day", "mean"),
            p25_veh=("Vehicles Per Day", lambda s: float(np.percentile(s, 25))),
            p75_veh=("Vehicles Per Day", lambda s: float(np.percentile(s, 75))),
            tmax=("MAX_TEMPERATURE", "mean"),
            tmin=("MIN_TEMPERATURE", "mean"),
            precip=("TOTAL_PRECIPITATION", "mean"),
        )
    )
    grid = pd.DataFrame({"ref_ds": ref_dates, "month": ref_dates.month, "day": ref_dates.day})
    grid = grid.merge(daily_agg, on=["month", "day"], how="left")

    # Circular 11-day seasonal smoothing for clean calendar heatmap display
    pad_n = 5
    padded = pd.concat([grid.iloc[-pad_n:], grid, grid.iloc[:pad_n]], ignore_index=True)
    for col in ("avg_veh", "p25_veh", "p75_veh", "tmax", "tmin", "precip"):
        padded[col] = padded[col].interpolate(limit_direction="both")
        padded[col] = padded[col].rolling(11, center=True, min_periods=1).mean()
    smoothed = padded.iloc[pad_n:-pad_n].reset_index(drop=True)

    daily_profile_365 = []
    for doy_idx, row in smoothed.iterrows():
        ds_r: pd.Timestamp = row["ref_ds"]
        daily_profile_365.append(
            {
                "month": int(row["month"]),
                "day": int(row["day"]),
                "doy": int(doy_idx + 1),
                "date_label": ds_r.strftime("%b %d"),
                "avg_vehicles": round(float(row["avg_veh"])),
                "p25_vehicles": round(float(row["p25_veh"])),
                "p75_vehicles": round(float(row["p75_veh"])),
                "temp_max_c": round(float(row["tmax"]), 1) if pd.notna(row["tmax"]) else None,
                "temp_min_c": round(float(row["tmin"]), 1) if pd.notna(row["tmin"]) else None,
                "precip_mm": round(float(row["precip"]), 1) if pd.notna(row["precip"]) else 0.0,
                "season_tag": _season_tag_for_md(int(row["month"]), int(row["day"])),
            }
        )

    # Day of Week profile (trailing 3 years)
    dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    dow_means = recent.groupby("Day of Week")["Vehicles Per Day"].mean().reindex(dow_order)
    dow_profile = [
        {
            "dow": d,
            "short": d[:3],
            "avg_vehicles": round(float(dow_means[d])) if pd.notna(dow_means[d]) else 0,
            "is_weekend": d in ("Saturday", "Sunday"),
        }
        for d in dow_order
    ]

    # Monthly profile across recent 3 years + monthly typical summary
    recent_years = sorted(recent["Year"].unique())[-3:]
    m_yr = (
        recent[recent["Year"].isin(recent_years)]
        .groupby(["Year", "Month"], as_index=False)["Vehicles Per Day"]
        .mean()
    )
    monthly_profile = [
        {
            "year": int(r["Year"]),
            "month": int(r["Month"]),
            "avg_vehicles": round(float(r["Vehicles Per Day"])),
        }
        for _, r in m_yr.iterrows()
    ]

    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    m_typ = (
        recent.groupby("month", as_index=False)
        .agg(
            avg_veh=("Vehicles Per Day", "mean"),
            tmax=("MAX_TEMPERATURE", "mean"),
            tmin=("MIN_TEMPERATURE", "mean"),
            precip=("TOTAL_PRECIPITATION", "mean"),
        )
        .sort_values("month")
    )
    monthly_typical = [
        {
            "month": int(r["month"]),
            "name": month_names[int(r["month"]) - 1],
            "avg_vehicles": round(float(r["avg_veh"])),
            "temp_max_c": round(float(r["tmax"]), 1) if pd.notna(r["tmax"]) else None,
            "temp_min_c": round(float(r["tmin"]), 1) if pd.notna(r["tmin"]) else None,
            "precip_mm": round(float(r["precip"]), 1) if pd.notna(r["precip"]) else 0.0,
        }
        for _, r in m_typ.iterrows()
    ]

    return {
        "daily_profile_365": daily_profile_365,
        "dow_profile": dow_profile,
        "monthly_profile": monthly_profile,
        "monthly_typical": monthly_typical,
    }


def build_forecast_payload(
    *,
    traffic_df: pd.DataFrame,
    exog_panel: pd.DataFrame,
    forecast_enriched: pd.DataFrame,
    driver_attributions: List[Dict],
    summary_metrics: pd.DataFrame,
    model_weights: Dict[str, float],
    cv_info: Dict[str, Any],
    freshness: Dict[str, Any],
    today_mt: pd.Timestamp,
) -> Dict[str, Any]:
    """Build verified JSON payload driving both the static web app and Streamlit UI."""
    weekly_insight, seasonal_insight = build_insights(traffic_df)
    upcoming_only = forecast_enriched[forecast_enriched["ds"] >= today_mt]
    conf_lbl, conf_det = get_conformal_confidence_summary(
        upcoming_only if not upcoming_only.empty else forecast_enriched,
        cv_info=cv_info,
    )

    # Last 45 days observed history for chart continuity
    ex = exog_panel.copy()
    ex.index = pd.to_datetime(ex.index).tz_localize(None).normalize()
    hist_tail = traffic_df.tail(45).copy()
    recent_history = []
    for _, r in hist_tail.iterrows():
        ds = pd.Timestamp(r["Date"]).normalize()
        er = ex.loc[ds] if ds in ex.index else {}
        recent_history.append(
            {
                "ds": ds.strftime("%Y-%m-%d"),
                "vehicles": round(float(r["Vehicles Per Day"])),
                "west_entrance": round(float(r["West_Entrance"])) if pd.notna(r.get("West_Entrance")) else None,
                "east_entrance": round(float(r["East_Entrance"])) if pd.notna(r.get("East_Entrance")) else None,
                "mountain_ave": round(float(r["Mountain_Ave"])) if pd.notna(r.get("Mountain_Ave")) else None,
                "is_weekend": bool(r["Is Weekend"]),
                "is_observed": bool(r.get("is_observed", True)),
                "temp_max_c": round(float(er["MAX_TEMPERATURE"]), 1) if "MAX_TEMPERATURE" in er and pd.notna(er["MAX_TEMPERATURE"]) else None,
                "temp_min_c": round(float(er["MIN_TEMPERATURE"]), 1) if "MIN_TEMPERATURE" in er and pd.notna(er["MIN_TEMPERATURE"]) else None,
                "precip_mm": round(float(er["TOTAL_PRECIPITATION"]), 1) if "TOTAL_PRECIPITATION" in er and pd.notna(er["TOTAL_PRECIPITATION"]) else 0.0,
            }
        )

    forecast_days = []
    fc_sorted = forecast_enriched.sort_values("ds").reset_index(drop=True)
    for i, row in fc_sorted.iterrows():
        ds = pd.Timestamp(row["ds"]).normalize()
        attr = driver_attributions[i] if i < len(driver_attributions) else {"drivers": []}
        models_dict = {m: round(float(row[m]), 1) for m in BASE_MODEL_NAMES if m in row and pd.notna(row[m])}
        forecast_days.append(
            {
                "ds": ds.strftime("%Y-%m-%d"),
                "dow": str(row["Day of Week"]),
                "dow_short": ds.strftime("%a"),
                "date_short": f"{ds.strftime('%b')} {ds.day}",
                "is_weekend": bool(row["Is Weekend"]),
                "is_bridge_gap": bool(ds < today_mt),
                "ensemble": round(float(row["Ensemble"])),
                "pi80_low": round(float(row["pi80_low"])),
                "pi80_high": round(float(row["pi80_high"])),
                "pi95_low": round(float(row["pi95_low"])),
                "pi95_high": round(float(row["pi95_high"])),
                "west_entrance": int(row["west_entrance"]),
                "east_entrance": int(row["east_entrance"]),
                "mountain_ave": int(row["mountain_ave"]),
                "west_share_pct": float(row["west_share_pct"]),
                "dow_seasonal_norm": round(float(row["dow_seasonal_norm"])),
                "vs_dow_norm_pct": round(float(row["vs_dow_norm_pct"]), 1),
                "overall_percentile": float(row["overall_percentile"]),
                "daytype_percentile": float(row["daytype_percentile"]),
                "signal_overall": str(row["signal_overall"]),
                "tone_overall": str(row["tone_overall"]),
                "signal_daytype": str(row["signal_daytype"]),
                "tone_daytype": str(row["tone_daytype"]),
                "traffic_score_overall": float(row["traffic_score_overall"]),
                "traffic_score_daytype": float(row["traffic_score_daytype"]),
                "weather_score": float(row["weather_score"]),
                "weather_label": str(row["weather_label"]),
                "weather_icon": str(row["weather_icon"]),
                "temp_max_c": row["temp_max_c"],
                "temp_min_c": row["temp_min_c"],
                "precip_prob_pct": row["precip_prob_pct"],
                "precip_mm": float(row["precip_mm"]),
                "snowfall_cm": float(row["snowfall_cm"]),
                "wind_max_kmh": row["wind_max_kmh"],
                "visit_score_quiet": float(row["visit_score_quiet"]),
                "visit_score_balanced": float(row["visit_score_balanced"]),
                "visit_score_weather": float(row["visit_score_weather"]),
                "visit_score_scoped_balanced": float(row["visit_score_scoped_balanced"]),
                "event_label": str(row.get("event_label", "") or ""),
                "driver_attributions": attr,
                "models": models_dict,
            }
        )

    upcoming_records = [d for d in forecast_days if not d["is_bridge_gap"]] or forecast_days
    payload = {
        "schema_version": "2.0.0",
        "generated_at_utc": freshness["generated_at_utc"],
        "generated_date_mt": today_mt.strftime("%Y-%m-%d"),
        "forecast_window": {
            "start": upcoming_records[0]["ds"],
            "end": upcoming_records[-1]["ds"],
            "n_days": len(upcoming_records),
            "total_horizon_days": len(forecast_days),
        },
        "data_freshness": freshness,
        "cv_benchmark": {
            **cv_info,
            "ensemble_weights": {k: round(float(v), 4) for k, v in model_weights.items()},
            "summary_metrics": summary_metrics.to_dict(orient="records"),
        },
        "insights": {
            "weekly_insight": weekly_insight,
            "seasonal_insight": seasonal_insight,
            "conformal_label": conf_lbl,
            "conformal_detail": conf_det,
        },
        "recent_history": recent_history,
        "forecast_days": forecast_days,
        "annual_planner": build_annual_planner_data(traffic_df, exog_panel),
    }
    return _json_safe(payload)


def _validate_forecast_outputs(
    forecast_df: pd.DataFrame,
    traffic_df: pd.DataFrame,
    today_mt: pd.Timestamp,
) -> None:
    """Enforce mathematical & production sanity assertions prior to persisting any artifact."""
    if forecast_df.empty:
        raise ValueError("Generated forecast DataFrame is empty.")
    for col in ["Ensemble", "pi80_low", "pi80_high", "pi95_low", "pi95_high", "west_entrance", "east_entrance"]:
        if col not in forecast_df.columns:
            raise KeyError(f"Missing required forecast output column: {col}")
        if forecast_df[col].isna().any():
            raise ValueError(f"Forecast column '{col}' contains NaN values.")

    if not ((forecast_df["pi95_low"] <= forecast_df["pi80_low"]).all() and
            (forecast_df["pi80_low"] <= forecast_df["Ensemble"]).all() and
            (forecast_df["Ensemble"] <= forecast_df["pi80_high"]).all() and
            (forecast_df["pi80_high"] <= forecast_df["pi95_high"]).all()):
        raise ValueError("Conformal prediction interval monotonicity violated (pi95_low <= pi80_low <= Ensemble <= pi80_high <= pi95_high).")

    recent_med = float(traffic_df["Vehicles Per Day"].tail(30).median())
    if (forecast_df["Ensemble"] < recent_med * 0.35).all():
        raise ValueError("Forecast median is implausibly collapsed relative to trailing 30d actuals.")

    if ((forecast_df["west_entrance"] + forecast_df["east_entrance"]) - forecast_df["Ensemble"].round()).abs().max() > 2.0:
        raise ValueError("Entrance disaggregation (west + east) diverged from Ensemble total.")

    upcoming = forecast_df[forecast_df["ds"] >= today_mt]
    if len(upcoming) < 15:
        raise ValueError(f"Forecast horizon must cover >= 15 future days from {today_mt.date()}, found {len(upcoming)}.")


def run_full_pipeline(
    *,
    refresh_web: bool = False,
    run_cv: bool = True,
    rebuild_site: bool = True,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
    outputs_dir: Path = OUTPUTS_DIR,
    site_dir: Path = SITE_DIR,
    root_predictions_csv: Path = ROOT_PREDICTIONS_CSV,
) -> Dict[str, Any]:
    """Execute full Banff forecasting, conformal calibration, export, and web build pipeline."""
    outputs_dir.mkdir(parents=True, exist_ok=True)
    today_mt = _calgary_today()

    if refresh_web:
        fetch_banff_traffic_tableau(refresh=True)
    traffic_df = load_daily_traffic()

    last_obs = pd.Timestamp(traffic_df["Date"].max()).normalize()
    target_end = max(today_mt + pd.Timedelta(days=15), last_obs + pd.Timedelta(days=horizon_days))
    needed_om_days = max(16, int((target_end - today_mt).days) + 1)

    exog_panel = build_daily_exogenous_panel(
        horizon_days=needed_om_days,
        cache_dir=CACHE_DIR,
        refresh=refresh_web,
    )
    future_index = pd.date_range(last_obs + pd.Timedelta(days=1), target_end, freq="D")

    cv_summary_csv = outputs_dir / "cv_summary_metrics.csv"
    cv_folds_csv = outputs_dir / "cv_fold_metrics.csv"
    cv_oof_csv = outputs_dir / "cv_oof_predictions.csv"
    cv_meta_json = outputs_dir / "cv_meta.json"

    if run_cv or not (cv_summary_csv.exists() and cv_oof_csv.exists() and cv_meta_json.exists()):
        logger.info("Running 12-fold multi-season expanding CV...")
        fold_metrics, summary_metrics, oof_predictions, model_weights, cv_info = run_expanding_cv(
            traffic_df, exog_panel, n_folds=12, horizon=DEFAULT_HORIZON_DAYS, step_days=60
        )
        summary_metrics.to_csv(cv_summary_csv, index=False)
        fold_metrics.to_csv(cv_folds_csv, index=False)
        oof_predictions.to_csv(cv_oof_csv, index=False)
        cv_meta_json.write_text(
            json.dumps({"model_weights": model_weights, "cv_info": cv_info}, indent=2),
            encoding="utf-8",
        )
    else:
        logger.info("Reusing cached expanding CV & conformal calibration from %s", outputs_dir)
        summary_metrics = pd.read_csv(cv_summary_csv)
        oof_predictions = pd.read_csv(cv_oof_csv)
        oof_predictions["ds"] = pd.to_datetime(oof_predictions["ds"]).dt.normalize()
        meta_loaded = json.loads(cv_meta_json.read_text(encoding="utf-8"))
        model_weights = meta_loaded["model_weights"]
        cv_info = meta_loaded["cv_info"]

    # Full refit on all complete historical observations through last_obs
    pred_raw, diag = fit_predict_daily_models(
        traffic_df=traffic_df,
        exog_panel=exog_panel,
        future_index=future_index,
        train_end=None,
        model_weights=model_weights,
    )
    fc_df = pred_raw.rename(columns={"Ensemble_Weighted": "Ensemble"}).copy()
    fc_df["ds"] = fc_df.index
    fc_df.index.name = None

    ci_df = compute_conformal_intervals(fc_df["Ensemble"], oof_predictions, model_name="Ensemble_Weighted")
    for col in ("pi80_low", "pi80_high", "pi95_low", "pi95_high"):
        fc_df[col] = ci_df[col].to_numpy(dtype=float)
    fc_df["yhat_lower"] = fc_df["pi80_low"]
    fc_df["yhat_upper"] = fc_df["pi80_high"]

    fc_df = predict_entrance_breakdown(traffic_df, fc_df)
    fc_enriched = enrich_forecast_analytics(fc_df, traffic_df, exog_panel)
    driver_attributions = compute_daily_driver_attributions(diag, fc_enriched)

    _validate_forecast_outputs(fc_enriched, traffic_df, today_mt)

    # Save enriched CSV & compatible root predictions.csv cleanly
    full_csv_df = fc_enriched.copy()
    full_csv_df.insert(0, "unique_id", "traffic")
    full_csv_df["ds"] = pd.to_datetime(full_csv_df["ds"]).dt.strftime("%Y-%m-%d")
    full_csv_df.to_csv(outputs_dir / "daily_forecast_full.csv", index=False)

    root_cols = [
        "unique_id",
        "ds",
        *BASE_MODEL_NAMES,
        "Ensemble",
        "yhat_lower",
        "yhat_upper",
        "pi80_low",
        "pi80_high",
        "pi95_low",
        "pi95_high",
        "west_entrance",
        "east_entrance",
        "mountain_ave",
        "event_label",
        "weather_label",
        "temp_max_c",
        "temp_min_c",
        "precip_prob_pct",
        "precip_mm",
        "snowfall_cm",
        "weather_score",
    ]
    root_out = full_csv_df[[c for c in root_cols if c in full_csv_df.columns]].copy()
    for c in [*BASE_MODEL_NAMES, "Ensemble", "yhat_lower", "yhat_upper", "pi80_low", "pi80_high", "pi95_low", "pi95_high"]:
        if c in root_out.columns:
            root_out[c] = root_out[c].round(2)
    root_out.to_csv(root_predictions_csv, index=False)

    freshness = summarize_data_freshness(traffic_df, exog_panel, cache_dir=CACHE_DIR)
    payload = build_forecast_payload(
        traffic_df=traffic_df,
        exog_panel=exog_panel,
        forecast_enriched=fc_enriched,
        driver_attributions=driver_attributions,
        summary_metrics=summary_metrics,
        model_weights=model_weights,
        cv_info=cv_info,
        freshness=freshness,
        today_mt=today_mt,
    )
    (outputs_dir / "forecast_payload.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if rebuild_site:
        build_static_site(payload, site_dir=site_dir)

    logger.info(
        "Pipeline completed cleanly: %s..%s (%d future days).",
        payload["forecast_window"]["start"],
        payload["forecast_window"]["end"],
        payload["forecast_window"]["n_days"],
    )
    return payload
