"""Leakage-free daily feature engineering: Alberta holidays, bridges, harmonics, weather, and macro/FX."""

from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import holidays
import numpy as np
import pandas as pd

from banff_planner.config import describe_wmo_code


@lru_cache(maxsize=16)
def _ab_holidays_dict(years_tuple: tuple) -> Dict[object, str]:
    """Return Alberta statutory holidays + Easter Monday for visitor modeling."""
    ca_ab = holidays.CA(subdiv="AB", years=list(years_tuple))
    result = {d: str(name) for d, name in ca_ab.items()}
    # Ensure Easter Monday (schools / federal government long weekend in AB) is marked
    for d, name in list(ca_ab.items()):
        if "Good Friday" in str(name):
            em = pd.Timestamp(d) + pd.Timedelta(days=3)
            result.setdefault(em.date(), "Easter Monday")
    return result


MAJOR_HOLIDAY_KEYWORDS = (
    "New Year",
    "Family Day",
    "Good Friday",
    "Victoria Day",
    "Canada Day",
    "Heritage Day",
    "Civic Holiday",
    "Labour Day",
    "Thanksgiving",
    "Christmas",
)


def compute_weather_comfort_score(
    month: int,
    temp_max: float,
    temp_min: float,
    precip_mm: float,
    precip_prob: Optional[float],
    wind_kmh: Optional[float],
    snowfall_cm: float,
    weathercode: Optional[float],
) -> float:
    """Compute a seasonal 0-100 tourist weather comfort score tailored to Banff, Alberta.

    Handles all 4 mountain seasons fairly: crisp winter/early-spring ski days (-8..+6 C)
    score well unless extreme polar cold or heavy rain/wind occurs, while May-Oct rewards
    mild dry clear conditions (12..25 C).
    """
    _, _, wmo_pen = describe_wmo_code(weathercode)
    score = 100.0 - 0.55 * wmo_pen

    # Thermal comfort depends on winter/ski vs warm/hiking season
    tmax = float(temp_max) if np.isfinite(temp_max) else 5.0
    tmin = float(temp_min) if np.isfinite(temp_min) else -5.0
    if month in (5, 6, 7, 8, 9, 10):
        if tmax < 12.0:
            score -= min(28.0, (12.0 - tmax) * 1.4)
        elif tmax > 27.0:
            score -= min(18.0, (tmax - 27.0) * 2.2)
        if tmin < -2.0:
            score -= min(12.0, (-2.0 - tmin) * 1.2)
    else:
        # Winter & ski season (Nov-Apr): crisp -10..+6 C is comfortable; extreme cold hurts
        if tmax < -12.0:
            score -= min(40.0, (-12.0 - tmax) * 2.6)
        if tmin < -20.0:
            score -= min(20.0, (-20.0 - tmin) * 1.5)
        if tmax > 10.0 and precip_mm > 2.0:
            score -= 8.0

    # Precipitation & wet snow penalties
    p_mm = max(0.0, float(precip_mm)) if np.isfinite(precip_mm) else 0.0
    sf_cm = max(0.0, float(snowfall_cm)) if np.isfinite(snowfall_cm) else 0.0
    is_rain = p_mm > 0.2 and (sf_cm < 0.3 * p_mm or tmax > 2.0)
    if is_rain:
        score -= min(32.0, np.log1p(p_mm) * 11.5)
    elif sf_cm > 0.0:
        # Fresh powder snow has smaller penalty than freezing rain unless storm (>8cm)
        score -= min(24.0, max(0.0, sf_cm - 2.0) * 2.0)

    if precip_prob is not None and np.isfinite(precip_prob):
        p_prob = float(np.clip(precip_prob, 0.0, 100.0))
        if p_prob > 35.0:
            score -= (p_prob - 35.0) * 0.22

    if wind_kmh is not None and np.isfinite(wind_kmh):
        w = float(max(0.0, wind_kmh))
        if w > 28.0:
            score -= min(18.0, (w - 28.0) * 0.65)

    return float(np.clip(round(score, 1), 5.0, 100.0))


def build_daily_calendar_features(index: pd.DatetimeIndex) -> Tuple[pd.DataFrame, pd.Series]:
    """Build leakage-free daily calendar features + human-readable holiday/event labels."""
    idx = pd.DatetimeIndex(index)
    years = tuple(sorted(set(idx.year.tolist()) | {idx.min().year - 1, idx.max().year + 1}))
    hol_dict = _ab_holidays_dict(years)
    hol_dates = set(hol_dict.keys())

    doy = np.asarray(idx.day_of_year, dtype=float)
    # Continuous annual phase accounting for leap years cleanly (365.25d)
    t_days = (idx - pd.Timestamp("2013-07-01")).days.to_numpy(dtype=float)
    wk_phase = 2.0 * np.pi * idx.day_of_week.to_numpy(dtype=float) / 7.0
    yr_phase = 2.0 * np.pi * (doy - 1.0) / 365.25

    data: Dict[str, np.ndarray] = {}
    for k in (1, 2, 3):
        data[f"wk_sin{k}"] = np.sin(k * wk_phase)
        data[f"wk_cos{k}"] = np.cos(k * wk_phase)
    for k in (1, 2, 3, 4):
        data[f"yr_sin{k}"] = np.sin(k * yr_phase)
        data[f"yr_cos{k}"] = np.cos(k * yr_phase)

    dow = idx.day_of_week.to_numpy()
    for d in range(7):
        data[f"dow_{d}"] = (dow == d).astype(float)

    data["is_friday"] = (dow == 4).astype(float)
    data["is_saturday"] = (dow == 5).astype(float)
    data["is_sunday"] = (dow == 6).astype(float)
    data["is_weekend"] = (dow >= 5).astype(float)

    # Structural time & COVID indicators (bounded / damped post-2022)
    t_years = t_days / 365.25
    data["t_years"] = t_years
    data["post_covid_level"] = (idx >= pd.Timestamp("2021-07-01")).astype(float)
    # Bounded post-COVID normalization ramp approaching 1.0 without unbounded explosion
    months_post_22 = np.maximum(0.0, (idx - pd.Timestamp("2022-01-01")).days.to_numpy(dtype=float) / 30.4375)
    data["post_2022_ramp"] = 1.0 - np.exp(-months_post_22 / 18.0)
    data["is_covid_acute"] = ((idx >= pd.Timestamp("2020-03-15")) & (idx <= pd.Timestamp("2020-06-30"))).astype(float)
    data["is_covid_era"] = ((idx >= pd.Timestamp("2020-03-15")) & (idx <= pd.Timestamp("2021-06-30"))).astype(float)

    # Alberta holidays, bridge weekends, and seasonal regimes
    n = len(idx)
    is_hol = np.zeros(n, dtype=float)
    is_major_hol = np.zeros(n, dtype=float)
    is_hol_eve = np.zeros(n, dtype=float)
    is_lw_fri = np.zeros(n, dtype=float)
    is_lw_sat = np.zeros(n, dtype=float)
    is_lw_sun = np.zeros(n, dtype=float)
    is_lw_mon = np.zeros(n, dtype=float)
    is_bridge = np.zeros(n, dtype=float)
    labels: List[str] = [""] * n

    for i, ts in enumerate(idx):
        d = ts.date()
        d_prev = (ts - pd.Timedelta(days=1)).date()
        d_next = (ts + pd.Timedelta(days=1)).date()
        d_plus2 = (ts + pd.Timedelta(days=2)).date()
        d_plus3 = (ts + pd.Timedelta(days=3)).date()
        d_minus2 = (ts - pd.Timedelta(days=2)).date()
        d_minus3 = (ts - pd.Timedelta(days=3)).date()
        wd = ts.weekday()

        tag_parts: List[str] = []
        if d in hol_dates:
            is_hol[i] = 1.0
            hname = hol_dict[d]
            tag_parts.append(hname)
            if any(k in hname for k in MAJOR_HOLIDAY_KEYWORDS):
                is_major_hol[i] = 1.0
            if wd == 0:  # Monday statutory holiday
                is_lw_mon[i] = 1.0
        if d_next in hol_dates:
            is_hol_eve[i] = 1.0
        # Friday preceding a Monday holiday (e.g. Victoria Day / Heritage Day / Labour Day / Thanksgiving)
        if wd == 4 and (d_plus3 in hol_dates):
            is_lw_fri[i] = 1.0
            tag_parts.append("Long weekend getaway")
        # Saturday of Monday-holiday or Good-Friday weekend
        if wd == 5 and (d_plus2 in hol_dates or d_prev in hol_dates):
            is_lw_sat[i] = 1.0
            if d not in hol_dates:
                tag_parts.append("Long weekend Saturday")
        # Sunday of Monday-holiday or Easter weekend (high day-trip & hotel stay volume)
        if wd == 6 and (d_next in hol_dates or d_minus2 in hol_dates):
            is_lw_sun[i] = 1.0
            if d not in hol_dates:
                tag_parts.append("Long weekend Sunday")
        # Weekday bridge adjacent to mid-week holiday (e.g. Fri after Thu holiday or Mon before Tue holiday)
        if wd == 4 and d_prev in hol_dates and d not in hol_dates:
            is_bridge[i] = 1.0
            tag_parts.append("Bridge day")
        if wd == 0 and d_next in hol_dates and d not in hol_dates:
            is_bridge[i] = 1.0
            tag_parts.append("Bridge day")

        # Seasonal tourism regimes
        m, day_num = ts.month, ts.day
        if (m == 12 and day_num >= 23) or (m == 1 and day_num <= 3):
            if not tag_parts:
                tag_parts.append("Holiday & Ski Break")
        elif (m == 9 and day_num >= 15) or (m == 10 and day_num <= 10):
            if not tag_parts:
                tag_parts.append("Larch Season Peak")
        elif m in (7, 8) and wd >= 5 and not tag_parts:
            tag_parts.append("Peak Summer Weekend")

        labels[i] = " · ".join(dict.fromkeys(tag_parts))

    data["is_ab_holiday"] = is_hol
    data["is_major_holiday"] = is_major_hol
    data["is_holiday_eve"] = is_hol_eve
    data["is_long_weekend_fri"] = is_lw_fri
    data["is_long_weekend_sat"] = is_lw_sat
    data["is_long_weekend_sun"] = is_lw_sun
    data["is_long_weekend_mon"] = is_lw_mon
    data["is_bridge_day"] = is_bridge
    data["is_long_weekend_any"] = np.maximum.reduce([is_lw_fri, is_lw_sat, is_lw_sun, is_lw_mon, is_hol])

    # Seasonal periods
    month = idx.month.to_numpy()
    day_m = idx.day.to_numpy()
    data["is_summer_break"] = np.isin(month, [7, 8]).astype(float)
    data["is_larch_season"] = (((month == 9) & (day_m >= 15)) | ((month == 10) & (day_m <= 10))).astype(float)
    data["is_christmas_week"] = (((month == 12) & (day_m >= 23)) | ((month == 1) & (day_m <= 3))).astype(float)
    data["is_nov_quiet"] = ((month == 11) & (is_hol == 0)).astype(float)

    cal_df = pd.DataFrame(data, index=idx)
    cal_df.index.name = "ds"
    label_series = pd.Series(labels, index=idx, name="event_label")
    return cal_df, label_series


def build_daily_feature_matrix(exog_panel: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Assemble full daily leakage-free exogenous feature matrix + event labels from exog_panel."""
    idx = pd.DatetimeIndex(exog_panel.index)
    cal, labels = build_daily_calendar_features(idx)
    feats = cal.copy()

    # Weather drivers
    feats["temp_mean"] = exog_panel["MEAN_TEMPERATURE"].astype(float)
    feats["temp_max"] = exog_panel["MAX_TEMPERATURE"].astype(float)
    feats["temp_min"] = exog_panel["MIN_TEMPERATURE"].astype(float)
    feats["temp_anom"] = exog_panel["TEMP_ANOM"].astype(float)
    feats["diurnal_range"] = (feats["temp_max"] - feats["temp_min"]).clip(lower=0.0)
    feats["log1p_precip"] = np.log1p(exog_panel["TOTAL_PRECIPITATION"].clip(lower=0.0).astype(float))
    feats["snow_depth_scaled"] = exog_panel["SNOW_ON_GROUND"].clip(lower=0.0).astype(float) / 100.0
    feats["snowfall_cm"] = exog_panel.get("fcst_snowfall_cm", pd.Series(0.0, index=idx)).clip(lower=0.0).astype(float)
    feats["is_freezing"] = (feats["temp_max"] <= 0.0).astype(float)
    feats["is_heavy_precip"] = (exog_panel["TOTAL_PRECIPITATION"].astype(float) >= 5.0).astype(float)

    # Physical weather-weekend interaction terms
    feats["temp_anom_x_weekend"] = feats["temp_anom"] * feats["is_weekend"]
    feats["precip_x_weekend"] = feats["log1p_precip"] * feats["is_weekend"]
    feats["temp_anom_x_summer"] = feats["temp_anom"] * feats["is_summer_break"]

    # Comfort score feature (summarizes non-linear temperature + rain + wind + snow suitability)
    months = idx.month.to_numpy()
    tmax_a = exog_panel["fcst_temp_max"].astype(float).to_numpy() if "fcst_temp_max" in exog_panel else feats["temp_max"].to_numpy()
    tmin_a = exog_panel["fcst_temp_min"].astype(float).to_numpy() if "fcst_temp_min" in exog_panel else feats["temp_min"].to_numpy()
    p_a = exog_panel["TOTAL_PRECIPITATION"].astype(float).to_numpy()
    p_prob_a = exog_panel["fcst_precip_prob"].to_numpy(dtype=float) if "fcst_precip_prob" in exog_panel else np.full(len(idx), np.nan)
    w_a = exog_panel["fcst_wind_max_kmh"].to_numpy(dtype=float) if "fcst_wind_max_kmh" in exog_panel else np.full(len(idx), np.nan)
    sf_a = feats["snowfall_cm"].to_numpy()
    wmo_a = exog_panel["fcst_weathercode"].to_numpy(dtype=float) if "fcst_weathercode" in exog_panel else np.full(len(idx), np.nan)

    comfort = np.array(
        [
            compute_weather_comfort_score(
                int(months[i]), tmax_a[i], tmin_a[i], p_a[i], p_prob_a[i], w_a[i], sf_a[i], wmo_a[i]
            )
            for i in range(len(idx))
        ],
        dtype=float,
    )
    feats["weather_comfort_score"] = comfort
    feats["comfort_scaled"] = (comfort - 75.0) / 25.0

    # Causal Macro, FX & Alberta Tourism regime indicators
    if "fx_90d_lag60" in exog_panel:
        feats["log_fx_90d_lag60"] = np.log(exog_panel["fx_90d_lag60"].clip(lower=0.5).astype(float))
    elif "FXUSDCAD" in exog_panel:
        feats["log_fx_90d_lag60"] = np.log(exog_panel["FXUSDCAD"].clip(lower=0.5).astype(float))

    if "fx_30d_lag14" in exog_panel:
        feats["log_fx_30d_lag14"] = np.log(exog_panel["fx_30d_lag14"].clip(lower=0.5).astype(float))

    if "AB_Unemp_lag45" in exog_panel:
        feats["ab_unemp_lag45"] = exog_panel["AB_Unemp_lag45"].astype(float)
    if "AB_Gas_Real_lag45" in exog_panel:
        feats["log_gas_real_lag45"] = np.log(exog_panel["AB_Gas_Real_lag45"].clip(lower=1.0).astype(float))
    if "AB_FoodSvc_Real_lag60" in exog_panel:
        feats["log_foodsvc_real_lag60"] = np.log(exog_panel["AB_FoodSvc_Real_lag60"].clip(lower=1.0).astype(float))
    if "Hwy1_YoY_lag45" in exog_panel:
        feats["log_hwy1_yoy_lag45"] = np.log(exog_panel["Hwy1_YoY_lag45"].clip(lower=0.25, upper=3.0).astype(float))
    if "AB_Visitors_Intl_lag75" in exog_panel:
        feats["log_visitors_intl_lag75"] = np.log1p(exog_panel["AB_Visitors_Intl_lag75"].clip(lower=0.0).astype(float))

    return feats, labels
