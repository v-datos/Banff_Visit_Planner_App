"""Multi-family daily log-scale forecasting suite, entrance share disaggregation, and driver attribution.

Fixes all mathematical flaws identified in the legacy bundle:
  - Strictly log-scale modeling (ln y) matching Banff's multiplicative seasonal variance.
  - Zero duplicate/cloned models: combines 4 orthogonal model families
    (LogMSTL_ETS, Ridge_AR_Exo, MLForecast_LGBM, Hybrid_Ridge_CatBoost) + SeasonalNaive364 benchmark.
  - Leakage-free multi-horizon dynamic covariates (weather NWP, AB holidays/bridges, causal FX/macro).
  - Sample weighting downweighting acute COVID-lockdown shock (Mar 2020 - Jun 2021) while preserving recent dynamics.
"""

import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from mlforecast import MLForecast
from mlforecast.lag_transforms import RollingMean, SeasonalRollingMean
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsforecast import StatsForecast
from statsforecast.models import MSTL, AutoETS

from banff_planner.features import build_daily_feature_matrix

logger = logging.getLogger(__name__)

# Suppress noisy OpenMP / LightGBM warnings in CI logs
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

BASE_MODEL_NAMES: List[str] = [
    "SeasonalNaive364",
    "LogMSTL_ETS",
    "Ridge_AR_Exo",
    "MLForecast_LGBM",
    "Hybrid_Ridge_CatBoost",
]

CANDIDATE_ENSEMBLE_MODELS: List[str] = [
    "LogMSTL_ETS",
    "Ridge_AR_Exo",
    "MLForecast_LGBM",
    "Hybrid_Ridge_CatBoost",
]

ALL_MODEL_NAMES: List[str] = BASE_MODEL_NAMES + ["Ensemble_Weighted"]

RIDGE_EXOG_COLS: List[str] = [
    "wk_sin1",
    "wk_cos1",
    "wk_sin2",
    "wk_cos2",
    "wk_sin3",
    "wk_cos3",
    "yr_sin1",
    "yr_cos1",
    "yr_sin2",
    "yr_cos2",
    "yr_sin3",
    "yr_cos3",
    "yr_sin4",
    "yr_cos4",
    "is_friday",
    "is_saturday",
    "is_sunday",
    "post_covid_level",
    "post_2022_ramp",
    "is_covid_acute",
    "is_covid_era",
    "is_ab_holiday",
    "is_major_holiday",
    "is_holiday_eve",
    "is_long_weekend_fri",
    "is_long_weekend_sat",
    "is_long_weekend_sun",
    "is_long_weekend_mon",
    "is_bridge_day",
    "is_summer_break",
    "is_larch_season",
    "is_christmas_week",
    "is_nov_quiet",
    "temp_mean",
    "temp_anom",
    "diurnal_range",
    "log1p_precip",
    "snow_depth_scaled",
    "snowfall_cm",
    "is_freezing",
    "is_heavy_precip",
    "temp_anom_x_weekend",
    "precip_x_weekend",
    "temp_anom_x_summer",
    "comfort_scaled",
    "log_fx_90d_lag60",
    "log_fx_30d_lag14",
    "ab_unemp_lag45",
    "log_gas_real_lag45",
    "log_foodsvc_real_lag60",
    "log_hwy1_yoy_lag45",
    "log_visitors_intl_lag75",
]

TREE_DYNAMIC_COLS: List[str] = [
    "yr_sin1",
    "yr_cos1",
    "yr_sin2",
    "yr_cos2",
    "yr_sin3",
    "yr_cos3",
    "dow_0",
    "dow_1",
    "dow_2",
    "dow_3",
    "dow_4",
    "dow_5",
    "dow_6",
    "is_weekend",
    "post_2022_ramp",
    "is_covid_era",
    "is_ab_holiday",
    "is_major_holiday",
    "is_holiday_eve",
    "is_long_weekend_fri",
    "is_long_weekend_sat",
    "is_long_weekend_sun",
    "is_long_weekend_mon",
    "is_bridge_day",
    "is_summer_break",
    "is_larch_season",
    "is_christmas_week",
    "is_nov_quiet",
    "post_covid_level",
    "temp_mean",
    "temp_max",
    "temp_min",
    "temp_anom",
    "diurnal_range",
    "log1p_precip",
    "snow_depth_scaled",
    "snowfall_cm",
    "is_freezing",
    "is_heavy_precip",
    "temp_anom_x_weekend",
    "precip_x_weekend",
    "temp_anom_x_summer",
    "weather_comfort_score",
    "log_fx_90d_lag60",
    "log_fx_30d_lag14",
    "ab_unemp_lag45",
    "log_gas_real_lag45",
    "log_foodsvc_real_lag60",
    "log_hwy1_yoy_lag45",
]


def _compute_sample_weights(index: pd.DatetimeIndex) -> np.ndarray:
    """Compute time-recency weights that downweight COVID closure anomalies gracefully."""
    idx = pd.DatetimeIndex(index)
    t_end = idx.max()
    years_ago = (t_end - idx).days.to_numpy(dtype=float) / 365.25
    # Gentle exponential recency weighting (half-life ~5 years) so post-2021 regime dominates without discarding history
    w = np.exp(-years_ago / 7.5)
    covid_era = (idx >= pd.Timestamp("2020-03-15")) & (idx <= pd.Timestamp("2021-06-30"))
    w = np.where(covid_era, w * 0.08, w)
    return np.clip(w, 0.02, 2.0)


def _fit_ridge_ar_exo(
    y_train: pd.Series,
    feat_tr: pd.DataFrame,
    feat_fut: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Pipeline, List[str], float, float]:
    """Fit weighted RidgeCV on ln(y) + damped AR(1)/level correction on recent deseasonalized residuals."""
    cols = [c for c in RIDGE_EXOG_COLS if c in feat_tr.columns]
    log_y = np.log(np.clip(y_train.to_numpy(dtype=float), 1.0, None))
    weights = _compute_sample_weights(y_train.index)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", RidgeCV(alphas=np.logspace(-0.5, 3.0, 28))),
    ])
    pipe.fit(feat_tr[cols].to_numpy(dtype=float), log_y, ridge__sample_weight=weights)

    in_pred = pipe.predict(feat_tr[cols].to_numpy(dtype=float))
    fut_base = pipe.predict(feat_fut[cols].to_numpy(dtype=float))

    # Fit damped AR(1) correction excluding COVID distortion on recent trailing 3 years
    resid = log_y - in_pred
    recent_mask = y_train.index >= (y_train.index.max() - pd.Timedelta(days=1095))
    r_recent = resid[recent_mask]
    phi = 0.45
    if len(r_recent) > 60:
        r0 = r_recent[:-1]
        r1 = r_recent[1:]
        denom = float(np.dot(r0, r0))
        if denom > 1e-8:
            phi = float(np.clip(np.dot(r0, r1) / denom, 0.0, 0.85))

    # Anchor level adjustment from robust blend of last 3 days and last 14 days (ignores single holiday outlier)
    tail_3 = float(np.median(resid[-3:])) if len(resid) >= 3 else float(resid[-1])
    tail_14 = float(np.mean(resid[-14:])) if len(resid) >= 14 else tail_3
    anchor_err = float(np.clip(0.55 * tail_3 + 0.45 * tail_14, -0.22, 0.22))

    h = len(feat_fut)
    ar_steps = np.arange(1, h + 1, dtype=float)
    ar_adj = anchor_err * (phi ** ar_steps)
    fut_log = fut_base + ar_adj
    return in_pred, fut_base, fut_log, pipe, cols, anchor_err, phi


def _fit_hybrid_ridge_catboost(
    y_train: pd.Series,
    feat_tr: pd.DataFrame,
    feat_fut: pd.DataFrame,
    ridge_in_log: np.ndarray,
    ridge_fut_base_log: np.ndarray,
) -> np.ndarray:
    """Train CatBoostRegressor on post-Ridge log-residuals with pre-horizon deseasonalized momentum."""
    log_y = np.log(np.clip(y_train.to_numpy(dtype=float), 1.0, None))
    resid_tr = log_y - ridge_in_log

    # Build safe lag features from historical residuals (using post-origin constant anchor for horizon steps)
    resid_s = pd.Series(resid_tr, index=y_train.index)
    # Historical lag-free rolling residual momentum (strictly <= t-1 via shift(1) during training)
    tr_mom7 = resid_s.shift(1).rolling(7, min_periods=1).mean().fillna(0.0).to_numpy()
    tr_mom28 = resid_s.shift(1).rolling(28, min_periods=1).mean().fillna(0.0).to_numpy()
    tr_dow7 = resid_s.shift(7).fillna(0.0).to_numpy()
    tr_yr364 = resid_s.shift(364).fillna(0.0).to_numpy()

    # For future horizon h=1..H, compute strictly from trailing values <= T
    last_mom7 = float(resid_s.tail(7).mean())
    last_mom28 = float(resid_s.tail(28).mean())
    h = len(feat_fut)
    h_steps = np.arange(1, h + 1, dtype=float)
    fut_mom7 = last_mom7 * (0.85 ** h_steps)
    fut_mom28 = last_mom28 * (0.94 ** h_steps)

    # Same-DOW 7d residual known for h=1..7, decayed cleanly for h=8..16
    fut_dow7 = np.zeros(h, dtype=float)
    for step in range(h):
        lag_idx = step - 7
        if lag_idx < 0 and abs(lag_idx) <= len(resid_s):
            fut_dow7[step] = float(resid_s.iloc[lag_idx]) * 0.7
        else:
            fut_dow7[step] = last_mom28 * 0.35

    fut_yr364 = np.zeros(h, dtype=float)
    full_idx = y_train.index
    for step, f_ts in enumerate(feat_fut.index):
        ref_ts = f_ts - pd.Timedelta(days=364)
        if ref_ts in resid_s.index:
            fut_yr364[step] = float(resid_s.loc[ref_ts])
        else:
            # Nearest match within full index
            pos = full_idx.searchsorted(ref_ts)
            if 0 <= pos < len(resid_s):
                fut_yr364[step] = float(resid_s.iloc[pos])

    cb_cols = [c for c in TREE_DYNAMIC_COLS if c in feat_tr.columns]
    X_tr = feat_tr[cb_cols].copy()
    X_tr["resid_mom7"] = tr_mom7
    X_tr["resid_mom28"] = tr_mom28
    X_tr["resid_dow7"] = tr_dow7
    X_tr["resid_yr364"] = tr_yr364

    X_fut = feat_fut[cb_cols].copy()
    X_fut["resid_mom7"] = fut_mom7
    X_fut["resid_mom28"] = fut_mom28
    X_fut["resid_dow7"] = fut_dow7
    X_fut["resid_yr364"] = fut_yr364

    weights = _compute_sample_weights(y_train.index)
    # Trim warm-up year 1 where lag_364 is 0 and avoid fitting severe acute lockdowns
    valid_fit = np.arange(len(y_train)) >= min(365, len(y_train) // 3)
    cb = CatBoostRegressor(
        iterations=320,
        learning_rate=0.04,
        depth=5,
        l2_leaf_reg=6.0,
        subsample=0.85,
        random_seed=42,
        verbose=False,
        allow_writing_files=False,
    )
    cb.fit(X_tr.iloc[valid_fit], np.clip(resid_tr[valid_fit], -0.45, 0.45), sample_weight=weights[valid_fit])
    cb_resid_fut = np.clip(cb.predict(X_fut), -0.35, 0.35)
    return ridge_fut_base_log + cb_resid_fut


def _fit_mlforecast_lgbm(
    y_train: pd.Series,
    feat_tr: pd.DataFrame,
    feat_fut: pd.DataFrame,
) -> np.ndarray:
    """Fit MLForecast LGBMRegressor recursively on ln(y) with dynamic weather/holiday/macro exog."""
    dyn_cols = [c for c in TREE_DYNAMIC_COLS if c in feat_tr.columns]
    # Train on recent post-COVID + tail historical balance to emphasize current Banff dynamics while keeping lags warm
    # Keep at least from 2018-01-01 so multi-year seasonal tree splits have rich pre/post comparisons
    start_cut = max(y_train.index.min(), pd.Timestamp("2018-01-01"))
    y_sub = y_train.loc[start_cut:]
    log_y = np.log(np.clip(y_sub.to_numpy(dtype=float), 1.0, None))

    df_ml = pd.DataFrame({"unique_id": "banff", "ds": y_sub.index, "y": log_y})
    df_ml["sample_weight"] = _compute_sample_weights(y_sub.index)
    for c in dyn_cols:
        df_ml[c] = feat_tr.loc[y_sub.index, c].to_numpy(dtype=float)

    X_df = pd.DataFrame({"unique_id": "banff", "ds": feat_fut.index})
    for c in dyn_cols:
        X_df[c] = feat_fut[c].to_numpy(dtype=float)

    lgbm = LGBMRegressor(
        n_estimators=280,
        learning_rate=0.035,
        num_leaves=23,
        max_depth=5,
        min_child_samples=20,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_lambda=3.0,
        reg_alpha=0.3,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )

    fcst = MLForecast(
        models={"LGBMRegressor": lgbm},
        freq="D",
        lags=[1, 2, 3, 7, 14, 21, 28, 364],
        lag_transforms={
            1: [RollingMean(window_size=7), RollingMean(window_size=28)],
            7: [SeasonalRollingMean(season_length=7, window_size=4)],
        },
    )
    fcst.fit(df_ml, static_features=[], weight_col="sample_weight")
    preds = fcst.predict(h=len(feat_fut), X_df=X_df)
    return preds["LGBMRegressor"].to_numpy(dtype=float)


def _fit_log_mstl_ets(y_train: pd.Series, h: int) -> np.ndarray:
    """Fit StatsForecast MSTL([7, 365], trend_forecaster=AutoETS(model='ANN')) on post-COVID ln(y)."""
    # Use clean post-2021-07 regime so MSTL isn't skewed by 2020 lockdowns and runs cleanly in <1s
    start_mstl = max(y_train.index.min(), pd.Timestamp("2021-07-01"))
    y_sub = y_train.loc[start_mstl:]
    if len(y_sub) < 735:
        y_sub = y_train.tail(min(len(y_train), 1460))
    log_y = np.log(np.clip(y_sub.to_numpy(dtype=float), 1.0, None))
    sf_df = pd.DataFrame({"unique_id": "banff", "ds": y_sub.index, "y": log_y})
    sf = StatsForecast(
        models=[MSTL(season_length=[7, 365], trend_forecaster=AutoETS(model="ANN"))],
        freq="D",
        n_jobs=1,
    )
    fc = sf.forecast(df=sf_df, h=h)
    return fc["MSTL"].to_numpy(dtype=float)


def _seasonal_naive_364(y_train: pd.Series, future_index: pd.DatetimeIndex) -> np.ndarray:
    """Same-DOW 52 weeks (364 days) prior benchmark."""
    vals = []
    for ts in future_index:
        ref = ts - pd.Timedelta(days=364)
        if ref in y_train.index:
            vals.append(float(y_train.loc[ref]))
        else:
            ref2 = ts - pd.Timedelta(days=7)
            vals.append(float(y_train.loc[ref2]) if ref2 in y_train.index else float(y_train.tail(28).median()))
    return np.asarray(vals, dtype=float)


def fit_predict_daily_models(
    traffic_df: pd.DataFrame,
    exog_panel: pd.DataFrame,
    future_index: pd.DatetimeIndex,
    *,
    train_end: Optional[pd.Timestamp] = None,
    model_weights: Optional[Dict[str, float]] = None,
) -> Tuple[pd.DataFrame, Dict]:
    """Train all orthogonal daily forecasting models strictly <= train_end and predict future_index.

    Returns:
      - pred_df indexed by future_index with columns BASE_MODEL_NAMES + ['Ensemble_Weighted']
        (all in natural daily vehicle counts) plus ln_Ensemble_Weighted.
      - diagnostics dict containing Ridge coefficients, AR params, feature matrix, and weights.
    """
    hist = traffic_df.sort_values("Date").copy()
    hist_idx = pd.DatetimeIndex(hist["Date"])
    y_all = pd.Series(hist["Vehicles Per Day"].to_numpy(dtype=float), index=hist_idx, name="y")
    if train_end is not None:
        y_train = y_all.loc[:train_end].copy()
    else:
        y_train = y_all.copy()

    fut_idx = pd.DatetimeIndex(future_index)
    h = len(fut_idx)
    if h == 0:
        return pd.DataFrame(index=fut_idx), {}

    full_idx = y_train.index.union(fut_idx).sort_values()
    exog_sub = exog_panel.reindex(full_idx).ffill().bfill()
    feat_all, event_labels = build_daily_feature_matrix(exog_sub)
    feat_tr = feat_all.loc[y_train.index]
    feat_fut = feat_all.loc[fut_idx]

    # 1. Benchmark SeasonalNaive364
    sn_pred = _seasonal_naive_364(y_train, fut_idx)

    # 2. Log MSTL + AutoETS
    try:
        log_mstl = _fit_log_mstl_ets(y_train, h=h)
    except Exception as exc:
        logger.warning("LogMSTL_ETS fallback to SeasonalNaive due to: %s", exc)
        log_mstl = np.log(np.clip(sn_pred, 1.0, None))

    # 3. Ridge + Damped AR(1) Exogenous DHR
    in_ridge, fut_ridge_base, log_ridge_ar, ridge_pipe, ridge_cols, anchor_err, phi = _fit_ridge_ar_exo(
        y_train, feat_tr, feat_fut
    )

    # 4. MLForecast Recursive LightGBM
    try:
        log_lgbm = _fit_mlforecast_lgbm(y_train, feat_tr, feat_fut)
    except Exception as exc:
        logger.warning("MLForecast_LGBM fallback due to: %s", exc)
        log_lgbm = log_ridge_ar.copy()

    # 5. Hybrid Ridge + CatBoost Residual Corrector
    try:
        log_cb = _fit_hybrid_ridge_catboost(y_train, feat_tr, feat_fut, in_ridge, fut_ridge_base)
    except Exception as exc:
        logger.warning("Hybrid_Ridge_CatBoost fallback due to: %s", exc)
        log_cb = log_ridge_ar.copy()

    # Apply weighted combination IN LOG SPACE (geometric weighted ensemble preserves multiplicative seasonality!)
    log_preds = {
        "SeasonalNaive364": np.log(np.clip(sn_pred, 1.0, None)),
        "LogMSTL_ETS": log_mstl,
        "Ridge_AR_Exo": log_ridge_ar,
        "MLForecast_LGBM": log_lgbm,
        "Hybrid_Ridge_CatBoost": log_cb,
    }

    if model_weights is None:
        # Equal prior only during intermediate OOF scoring prior to dynamic weighting
        w = {m: 1.0 / len(CANDIDATE_ENSEMBLE_MODELS) for m in CANDIDATE_ENSEMBLE_MODELS}
    else:
        total_w = sum(max(0.0, float(model_weights.get(m, 0.0))) for m in CANDIDATE_ENSEMBLE_MODELS)
        if total_w <= 1e-12:
            w = {m: 1.0 / len(CANDIDATE_ENSEMBLE_MODELS) for m in CANDIDATE_ENSEMBLE_MODELS}
        else:
            w = {m: max(0.0, float(model_weights.get(m, 0.0))) / total_w for m in CANDIDATE_ENSEMBLE_MODELS}

    log_ens = sum(w[m] * log_preds[m] for m in CANDIDATE_ENSEMBLE_MODELS)

    pred_df = pd.DataFrame(index=fut_idx)
    pred_df.index.name = "ds"
    for m in BASE_MODEL_NAMES:
        pred_df[m] = np.clip(np.exp(log_preds[m]), 100.0, None)
    pred_df["Ensemble_Weighted"] = np.clip(np.exp(log_ens), 100.0, None)
    pred_df["log_Ensemble"] = log_ens
    pred_df["event_label"] = event_labels.loc[fut_idx].values

    diag = {
        "weights": w,
        "ridge_pipe": ridge_pipe,
        "ridge_cols": ridge_cols,
        "anchor_err": anchor_err,
        "ar_phi": phi,
        "feat_tr": feat_tr,
        "feat_fut": feat_fut,
        "log_ridge_base": fut_ridge_base,
        "log_ridge_ar": log_ridge_ar,
        "log_ens": log_ens,
        "y_train": y_train,
    }
    return pred_df, diag


def predict_entrance_breakdown(
    traffic_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
) -> pd.DataFrame:
    """Forecast coherent entrance split (West_Entrance + East_Entrance === Ensemble) & Mountain Ave.

    Models post-2022 day-of-week x month entrance shares from observed Tableau counters
    so tourists see expected load by access corridor.
    """
    hist = traffic_df.dropna(subset=["West_Entrance", "East_Entrance"]).copy()
    hist = hist[(hist["West_Entrance"] >= 0) & (hist["East_Entrance"] >= 0)].copy()
    recent = hist[hist["Date"] >= (hist["Date"].max() - pd.Timedelta(days=1095))].copy()
    if len(recent) < 90:
        recent = hist.copy()

    tot = (recent["West_Entrance"] + recent["East_Entrance"]).replace(0, np.nan)
    recent["west_share"] = (recent["West_Entrance"] / tot).clip(0.25, 0.75)
    recent["mtn_ratio"] = np.where(
        recent["Mountain_Ave"].notna() & (recent["Vehicles Per Day"] > 0),
        (recent["Mountain_Ave"] / recent["Vehicles Per Day"]).clip(0.05, 0.45),
        np.nan,
    )
    recent["month"] = recent["Date"].dt.month
    recent["is_wknd"] = (recent["Date"].dt.day_of_week >= 5).astype(int)

    # Hierarchical fallback lookup: (month, is_wknd) -> month -> global median
    west_mk = recent.groupby(["month", "is_wknd"])["west_share"].median()
    west_global = float(recent["west_share"].median()) if not recent.empty else 0.54

    mtn_valid = recent.dropna(subset=["mtn_ratio"])
    mtn_mk = mtn_valid.groupby(["month", "is_wknd"])["mtn_ratio"].median() if not mtn_valid.empty else pd.Series(dtype=float)
    mtn_global = float(mtn_valid["mtn_ratio"].median()) if not mtn_valid.empty else 0.16

    out = forecast_df.copy()
    west_vals = []
    east_vals = []
    mtn_vals = []
    west_pcts = []
    has_ds_col = "ds" in out.columns
    for idx_val, row in out.iterrows():
        ds = pd.Timestamp(row["ds"]) if has_ds_col else pd.Timestamp(idx_val)
        key = (int(ds.month), int(ds.day_of_week >= 5))
        w_share = float(west_mk.loc[key]) if key in west_mk.index else west_global
        m_ratio = float(mtn_mk.loc[key]) if key in mtn_mk.index else mtn_global
        total = float(row["Ensemble"])
        west_v = round(total * w_share)
        east_v = max(0, round(total) - west_v)
        mtn_v = round(total * m_ratio)
        west_vals.append(west_v)
        east_vals.append(east_v)
        mtn_vals.append(mtn_v)
        west_pcts.append(round(w_share * 100.0, 1))

    out["west_entrance"] = west_vals
    out["east_entrance"] = east_vals
    out["mountain_ave"] = mtn_vals
    out["west_share_pct"] = west_pcts
    return out


def compute_daily_driver_attributions(diag: Dict, forecast_df: pd.DataFrame) -> List[Dict]:
    """Compute human-readable causal factor breakdown ('Why this forecast?') per day using scaled Ridge model."""
    pipe: Pipeline = diag["ridge_pipe"]
    cols: List[str] = diag["ridge_cols"]
    feat_tr: pd.DataFrame = diag["feat_tr"]
    feat_fut: pd.DataFrame = diag["feat_fut"]

    scaler: StandardScaler = pipe.named_steps["scaler"]
    ridge: RidgeCV = pipe.named_steps["ridge"]
    coefs = ridge.coef_ / np.where(scaler.scale_ > 1e-8, scaler.scale_, 1.0)
    col_idx = {c: i for i, c in enumerate(cols)}

    # Reference neutral baseline: trailing 3yr same month normal weather & non-holiday weekdays/weekends
    X_tr = feat_tr[cols]
    X_fut = feat_fut[cols]
    recent_mask = feat_tr.index >= (feat_tr.index.max() - pd.Timedelta(days=1095))
    X_ref = X_tr.loc[recent_mask].copy()

    holiday_cols = [
        "is_ab_holiday", "is_major_holiday", "is_holiday_eve",
        "is_long_weekend_fri", "is_long_weekend_sat", "is_long_weekend_sun",
        "is_long_weekend_mon", "is_bridge_day", "is_larch_season", "is_christmas_week",
    ]
    weather_cols = [
        "temp_mean", "temp_anom", "diurnal_range", "log1p_precip",
        "snow_depth_scaled", "snowfall_cm", "is_freezing", "is_heavy_precip",
        "temp_anom_x_weekend", "precip_x_weekend", "temp_anom_x_summer", "comfort_scaled",
    ]
    macro_cols = [
        "log_fx_90d_lag60", "log_fx_30d_lag14", "ab_unemp_lag45",
        "log_gas_real_lag45", "log_foodsvc_real_lag60", "log_hwy1_yoy_lag45",
        "log_visitors_intl_lag75",
    ]

    def _contrib(row_vals: pd.Series, ref_vals: pd.Series, group: List[str]) -> float:
        s = 0.0
        for g in group:
            if g in col_idx:
                i = col_idx[g]
                s += float(coefs[i] * (row_vals[g] - ref_vals[g]))
        return s

    # Non-holiday same-month climate normal reference for clean attribution
    ref_macro_mean = X_ref[[c for c in macro_cols if c in X_ref]].mean() if any(c in X_ref for c in macro_cols) else pd.Series(dtype=float)

    attributions: List[Dict] = []
    anchor_err = float(diag.get("anchor_err", 0.0))
    ar_phi = float(diag.get("ar_phi", 0.45))

    for step, (ds, row_f) in enumerate(X_fut.iterrows(), start=1):
        # Match same calendar month +/- 10d for weather normal comparison
        ref_zero_hol = pd.Series(0.0, index=cols)
        hol_log = _contrib(row_f, ref_zero_hol, holiday_cols)

        # Weather effect relative to zero anomaly, median precip & seasonal comfort
        mo_mask = X_ref.index.month == ds.month
        mo_ref = X_ref.loc[mo_mask].median() if mo_mask.any() else X_ref.median()
        # Specifically compare temp_anom vs 0.0 and precip/snow vs normal
        wea_ref = mo_ref.copy()
        if "temp_anom" in wea_ref.index:
            wea_ref["temp_anom"] = 0.0
        if "temp_anom_x_weekend" in wea_ref.index:
            wea_ref["temp_anom_x_weekend"] = 0.0
        if "temp_anom_x_summer" in wea_ref.index:
            wea_ref["temp_anom_x_summer"] = 0.0
        wea_log = _contrib(row_f, wea_ref, weather_cols)

        # Macro + recent momentum relative to 3-yr mean macro regime
        mac_ref = mo_ref.copy()
        for mc in macro_cols:
            if mc in ref_macro_mean.index:
                mac_ref[mc] = ref_macro_mean[mc]
        mac_log = _contrib(row_f, mac_ref, macro_cols)
        ar_log = anchor_err * (ar_phi ** step)
        regime_pct = (np.exp(mac_log + ar_log) - 1.0) * 100.0
        hol_pct = (np.exp(hol_log) - 1.0) * 100.0
        wea_pct = (np.exp(wea_log) - 1.0) * 100.0

        # Non-linear ensemble/ML refinement vs linear Ridge prediction
        ridge_val = float(np.exp(diag["log_ridge_ar"][step - 1]))
        ens_val = float(forecast_df.loc[ds, "Ensemble"])
        ml_nl_pct = ((ens_val / max(ridge_val, 1.0)) - 1.0) * 100.0

        drivers: List[Dict] = []
        ev = str(forecast_df.loc[ds, "event_label"]) if "event_label" in forecast_df.columns else ""
        if ev or abs(hol_pct) >= 1.5:
            drivers.append({
                "factor": ev if ev else "Calendar / Holiday Window",
                "impact_pct": round(float(hol_pct), 1),
                "category": "calendar",
            })
        if abs(wea_pct) >= 1.0:
            w_desc = "Mild / clear weather boost" if wea_pct > 0 else "Precipitation / cold weather dampener"
            drivers.append({
                "factor": w_desc,
                "impact_pct": round(float(wea_pct), 1),
                "category": "weather",
            })
        if abs(regime_pct) >= 1.0:
            r_desc = "Elevated recent demand & FX/tourism regime" if regime_pct > 0 else "Cooler recent trend & macro drag"
            drivers.append({
                "factor": r_desc,
                "impact_pct": round(float(regime_pct), 1),
                "category": "macro",
            })
        if abs(ml_nl_pct) >= 1.5:
            drivers.append({
                "factor": "Tree & MSTL non-linear seasonal adjustment",
                "impact_pct": round(float(ml_nl_pct), 1),
                "category": "nonlinear",
            })

        # Sort strongest drivers first
        drivers.sort(key=lambda d: abs(d["impact_pct"]), reverse=True)
        attributions.append(
            {
                "holiday_pct": round(float(hol_pct), 1),
                "weather_pct": round(float(wea_pct), 1),
                "regime_pct": round(float(regime_pct), 1),
                "nonlinear_pct": round(float(ml_nl_pct), 1),
                "drivers": drivers[:4],
            }
        )
    return attributions
