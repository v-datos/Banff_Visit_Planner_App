"""Mathematical & statistical tests for multi-family forecasting, non-cloning, conformal PIs, and entrance split."""

import numpy as np
import pandas as pd

from banff_planner.evaluation import compute_conformal_intervals
from banff_planner.features import build_daily_calendar_features, compute_weather_comfort_score
from banff_planner.models import (
    BASE_MODEL_NAMES,
    compute_daily_driver_attributions,
    fit_predict_daily_models,
    predict_entrance_breakdown,
)


def _synthetic_traffic_and_exog(n_days: int = 1150, horizon: int = 16):
    dates = pd.date_range("2023-07-01", periods=n_days + horizon, freq="D")
    rng = np.random.default_rng(7)
    doy = dates.day_of_year.to_numpy(float)
    wknd = (dates.day_of_week >= 5).astype(float)
    temp_mean = 4.0 + 13.0 * np.sin(2.0 * np.pi * (doy - 110) / 365.25) + rng.normal(0, 1.5, len(dates))
    temp_max = temp_mean + 5.5
    temp_min = temp_mean - 5.5
    precip = rng.exponential(1.2, len(dates))
    snow_on_ground = np.where(np.isin(dates.month, [5, 6, 7, 8, 9]) | (temp_mean > 2.5), 0.0, 12.0)

    exog = pd.DataFrame(
        {
            "MEAN_TEMPERATURE": temp_mean,
            "MAX_TEMPERATURE": temp_max,
            "MIN_TEMPERATURE": temp_min,
            "TOTAL_PRECIPITATION": precip,
            "SNOW_ON_GROUND": snow_on_ground,
            "TEMP_ANOM": temp_mean - 4.0,
            "fcst_temp_max": temp_max,
            "fcst_temp_min": temp_min,
            "fcst_temp_mean": temp_mean,
            "fcst_precip_sum": precip,
            "fcst_precip_prob": np.clip(precip * 15.0, 0, 95),
            "fcst_snowfall_cm": np.where(temp_max < 1.0, precip * 0.4, 0.0),
            "fcst_wind_max_kmh": np.full(len(dates), 16.0),
            "fcst_weathercode": np.where(precip > 3.0, 61.0, 1.0),
            "fx_90d_lag60": np.full(len(dates), 1.37),
            "fx_30d_lag14": np.full(len(dates), 1.38),
            "AB_Unemp_lag45": np.full(len(dates), 6.8),
            "AB_Gas_Real_lag45": np.full(len(dates), 112.0),
            "AB_FoodSvc_Real_lag60": np.full(len(dates), 7800.0),
            "Hwy1_YoY_lag45": np.full(len(dates), 1.02),
            "AB_Visitors_Intl_lag75": np.full(len(dates), 90000.0),
        },
        index=dates,
    )

    hist_dates = dates[:n_days]
    signal = np.exp(
        9.45
        + 0.35 * np.sin(2.0 * np.pi * (doy[:n_days] - 110) / 365.25)
        + 0.28 * wknd[:n_days]
        + 0.012 * (temp_mean[:n_days] - 4.0)
    )
    vols = signal * rng.lognormal(0.0, 0.04, n_days)
    west = np.round(vols * 0.54)
    east = np.round(vols) - west
    traffic = pd.DataFrame(
        {
            "Date": hist_dates,
            "Vehicles Per Day": np.round(vols),
            "West_Entrance": west,
            "East_Entrance": east,
            "Mountain_Ave": np.round(vols * 0.16),
            "is_observed": True,
        }
    )
    fut_idx = dates[n_days:]
    return traffic, exog, fut_idx


def test_models_are_truly_distinct_and_entrance_sums_are_coherent():
    """Confirm no cloned/identical model predictions exist and West + East === Ensemble."""
    traffic, exog, fut_idx = _synthetic_traffic_and_exog()
    pred_df, diag = fit_predict_daily_models(traffic, exog, fut_idx)

    assert len(pred_df) == len(fut_idx)
    for col in BASE_MODEL_NAMES + ["Ensemble_Weighted"]:
        assert col in pred_df.columns
        assert pred_df[col].isna().sum() == 0
        assert (pred_df[col] > 1000.0).all()

    # Check every pair of candidate ensemble models differs meaningfully (zero clones like legacy predictions.csv)
    cands = ["LogMSTL_ETS", "Ridge_AR_Exo", "MLForecast_LGBM", "Hybrid_Ridge_CatBoost"]
    for i in range(len(cands)):
        for j in range(i + 1, len(cands)):
            max_diff = float(np.max(np.abs(pred_df[cands[i]].to_numpy() - pred_df[cands[j]].to_numpy())))
            assert max_diff > 10.0, f"{cands[i]} and {cands[j]} are nearly identical clones!"

    # Test coherent entrance breakdown & driver attributions
    fc = pred_df.rename(columns={"Ensemble_Weighted": "Ensemble"}).copy()
    fc["ds"] = fc.index
    fc = predict_entrance_breakdown(traffic, fc)
    assert ((fc["west_entrance"] + fc["east_entrance"]) - fc["Ensemble"].round()).abs().max() <= 1.0
    assert (fc["mountain_ave"] > 0).all()

    drivers = compute_daily_driver_attributions(diag, fc)
    assert len(drivers) == len(fut_idx)
    assert all("holiday_pct" in d and "weather_pct" in d for d in drivers)


def test_conformal_intervals_are_monotonic_and_calibrated():
    """Verify pi95_low <= pi80_low <= point <= pi80_high <= pi95_high across horizons."""
    fut_idx = pd.date_range("2026-11-13", periods=16, freq="D")
    pts = pd.Series(np.linspace(12000.0, 15500.0, 16), index=fut_idx)

    # Synthetic calibrated OOF residuals matching ~7% weekday, ~10% weekend noise
    rng = np.random.default_rng(123)
    oof_rows = []
    for k in range(12):
        ds_fold = pd.date_range(pd.Timestamp("2025-01-01") + pd.Timedelta(days=30 * k), periods=16, freq="D")
        for step, ts in enumerate(ds_fold, 1):
            sigma = 0.09 if ts.day_of_week >= 5 else 0.06
            y_pred = 14000.0
            y_true = y_pred * float(np.exp(rng.normal(0.0, sigma)))
            oof_rows.append(
                {
                    "ds": ts,
                    "fold": f"F{k}",
                    "horizon_step": step,
                    "model": "Ensemble_Weighted",
                    "y_true": y_true,
                    "y_pred": y_pred,
                }
            )
    oof_df = pd.DataFrame(oof_rows)
    pi = compute_conformal_intervals(pts, oof_df, model_name="Ensemble_Weighted")

    assert (pi["pi95_low"] <= pi["pi80_low"]).all()
    assert (pi["pi80_low"] <= pi["point"]).all()
    assert (pi["point"] <= pi["pi80_high"]).all()
    assert (pi["pi80_high"] <= pi["pi95_high"]).all()


def test_alberta_holiday_bridges_and_weather_comfort_score():
    """Verify Alberta Remembrance Day / Thanksgiving bridges and 4-season weather comfort logic."""
    idx = pd.to_datetime(["2026-07-11", "2026-10-12", "2026-11-11"])
    cal, labels = build_daily_calendar_features(idx)
    assert cal.loc[pd.Timestamp("2026-10-12"), "is_long_weekend_mon"] == 1.0  # Thanksgiving Monday
    assert cal.loc[pd.Timestamp("2026-11-11"), "is_ab_holiday"] == 1.0  # Remembrance Day
    assert "Remembrance Day" in labels.loc[pd.Timestamp("2026-11-11")]

    nice_summer = compute_weather_comfort_score(7, 22.0, 9.0, 0.0, 5.0, 12.0, 0.0, 1.0)
    storm_summer = compute_weather_comfort_score(7, 11.0, 4.0, 22.0, 95.0, 45.0, 0.0, 65.0)
    nice_winter = compute_weather_comfort_score(1, -2.0, -11.0, 0.0, 10.0, 10.0, 0.0, 0.0)
    polar_cold = compute_weather_comfort_score(1, -26.0, -34.0, 0.0, 15.0, 38.0, 0.0, 3.0)

    assert nice_summer > 90.0
    assert nice_summer > storm_summer + 40.0
    assert nice_winter > polar_cold + 35.0
