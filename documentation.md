# Banff Visit Planner — Technical & Mathematical Documentation

Comprehensive architecture, mathematical specification, data engineering safeguards, and deployment guide for **Banff Visit Planner v2.0**.

---

## 1. System Architecture

The repository splits into two cleanly decoupled layers so end-user apps never bundle heavyweight ML training packages:

1. **Automated Forecasting & Web Build Pipeline (`src/banff_planner/`)**
   - Executed daily by GitHub Actions (`.github/workflows/update_data.yml`, cron `0 11 * * *`) or manually via `banff-planner`.
   - Fetches and validates traffic counters, weather history/NWP, exchange rates, StatCan macro indicators, and Alberta park/highway counts (**strictly excluding municipal Zero Waste datasets**).
   - Evaluates 4 orthogonal log-scale forecasting families + a seasonal naive benchmark across 12 multi-season expanding folds (`n_oof = 192` days), calibrates **split-conformal 80% & 95% prediction intervals**, exports enriched CSV/JSON deliverables to `outputs/` & `predictions.csv`, and compiles the self-contained static web app to `site/index.html`.

2. **Tourist Frontends (`site/` & `traffic_app.py`)**
   - **Primary Static Web App (`site/index.html`):** Zero runtime backend, inline JSON payload (`#banff-payload-data`), HiDPI HTML5 Canvas visualization, light/dark alpine themes, URL `?view=planner|annual|science` support, interactive 365-day seasonal heatmap, and isolated logic module (`window.BanffPlannerLogic`) cross-verified under Node.js for exact parity with `banff_planner.analytics`.
   - **Streamlit Companion (`traffic_app.py`):** Uses strictly lightweight dependencies (`requirements.txt`: `streamlit>=1.44`, `pandas`, `plotly`, `numpy`, `holidays`, `requests`) and reads precomputed deliverables (`outputs/forecast_payload.json` + `predictions.csv` + `TW Traffic _data.csv`).

---

## 2. Mathematical & Statistical Fixes Implemented Over Legacy v1

| Legacy v1 Defect | v2.0 Resolution | Module |
| :--- | :--- | :--- |
| **Cloned neural predictions** (3–4 columns near-identical, fake zero-spread agreement) | 4 genuinely orthogonal model families (`LogMSTL_ETS`, `Ridge_AR_Exo`, `MLForecast_LGBM`, `Hybrid_Ridge_CatBoost`) with automated non-cloning test | [`models.py`](src/banff_planner/models.py) |
| **Static pretrained `modelos_exo` drift & additive scale mismatch** | Daily online refit strictly in **log scale ($\ln y$)** matching Banff's multiplicative seasonal amplitude, downweighting 2020–2021 COVID closure shocks cleanly | [`models.py`](src/banff_planner/models.py) |
| **Fake uncertainty bands** (min/max of correlated ensemble members ~1–4%) | Stratified **Split-Conformal Prediction Intervals** (`compute_conformal_intervals`) calibrated separately by weekday/weekend volatility and horizon step from 192 OOF CV residuals | [`evaluation.py`](src/banff_planner/evaluation.py) |
| **Weekend vs. weekday confounding in signals** | Dual evaluation: **`overall_percentile`** (±10d all-day window) AND **`daytype_percentile`** (strictly Saturdays/Sundays vs. historical weekends, weekdays vs. weekdays) | [`analytics.py`](src/banff_planner/analytics.py) |
| **Non-contiguous window bug** (8-day weekend gap in 3-day rolling average) | Strictly verified calendar contiguity `(end_ds - start_ds).days == (actual_days - 1)` in `compute_best_window()` | [`analytics.py`](src/banff_planner/analytics.py) |
| **Biased recent trend comparisons** under active weekday/weekend filters | Matched day-type trailing window + comparison against trailing 3-year same-DOW seasonal normal (`get_unbiased_trend_message()`) | [`analytics.py`](src/banff_planner/analytics.py) |
| **Tableau extraction risks** (overlapping sub-tables / 4× aggregation / partial tail / nullable Counter 3) | Explicit deduplication on `(Counter, TimeStamp)`, `>= 23h/day` completeness threshold, `Total / (West + East) <= 1.5` 4× guard, automatic partial-day trimming, nullable `Mountain_Ave` | [`exogenous.py`](src/banff_planner/exogenous.py), [`data_loader.py`](src/banff_planner/data_loader.py) |
| **Positional StatCan JSON mapping & broken Alberta FX splicing** | Integer `vectorId`-keyed StatCan parsing, Valet `FXUSDCAD` + legacy `IEXE0101` pre-2017 splice, Open-Meteo ERA5->BANFF CS bias calibration | [`exogenous.py`](src/banff_planner/exogenous.py) |

---

## 3. Exogenous Data Sources & Causal Lag Hygiene

Every exogenous indicator is lagged conservatively to prevent lookahead leakage during both cross-validation and live forecasting:

| Source | Endpoint / Feed | Variables | Causal Lag / Usage |
| :--- | :--- | :--- | :--- |
| **Town of Banff Traffic** | Tableau Public `.twb` → `.hyper` | `Vehicles Per Day`, `West_Entrance` (Counter 2), `East_Entrance` (Counter 1), `Mountain_Ave` (Counter 3) | Target `y` + coherent entrance share lookup `(month, is_weekend)` |
| **ECCC Climate Archive** | `api.weather.gc.ca` (`BANFF CS`, `3050519`) | Daily `MAX/MIN/MEAN_TEMPERATURE`, `TOTAL_PRECIPITATION`, `SNOW_ON_GROUND` | Ground-truth alpine weather (`1995–present`) |
| **Open-Meteo ERA5 & 16d NWP** | `archive-api.open-meteo.com`, `api.open-meteo.com` | Missing history gap-fill (monthly additive/ratio bias-calibrated against `BANFF CS`) + 16-day forecast (`temp`, `precip`, `precip_prob`, `snowfall`, `wind`, `WMO weathercode`) | Drives weather comfort score, thermal anomaly interactions & weather model terms |
| **Bank of Canada Valet** | `FXUSDCAD` + `IEXE0101` | Daily USD/CAD rate | Causal trailing rolling windows lagged 14d & 60d |
| **Statistics Canada WDS** | `getDataFromVectorsAndLatestNPeriods` (15 `STATCAN_VECTORS`) | AB CPI All/Food/Gasoline/Energy, AB Unemployment Rate, Wage, Food Services, Retail Trade, Real GDP, US/Overseas Visitors | Causal forward-filled publication lags (45d–75d) |
| **Alberta Economic Dashboard** | `api.economicdata.alberta.ca` (`Highway_count`, `NationalParkAttendance`) | Hwy 1 Banff Park Gates counts & monthly Park Attendance | YoY tourism momentum lagged 45d |
| **Alberta Calendar** | `holidays.CA(subdiv="AB")` + Easter Monday | Statutory holidays, long-weekend Fri/Sat/Sun/Mon, midweek bridge days, Larch Season (Sep 15–Oct 10), Summer Break, Christmas Week | Deterministic future schedule covariates + UI event badges |

*Zero Waste governance:* Tested automatically in `tests/test_data_ingestion.py::test_no_zero_waste_data_anywhere`.

---

## 4. Forecasting & Conformal Uncertainty Specification

1. **Models Combined (`CANDIDATE_ENSEMBLE_MODELS`)**:
   - `LogMSTL_ETS`: Multi-seasonal `[7, 365]` MSTL decomposition with `AutoETS(model="ANN")` trend forecaster fitted on post-July 2021 $\ln y$.
   - `Ridge_AR_Exo`: Sample-weighted `RidgeCV` on 52 engineered exogenous + Fourier (`wk=1..3`, `yr=1..4`) + weather-weekend interaction features, paired with a damped AR(1) anchor correction (`max |anchor| <= 0.22`).
   - `MLForecast_LGBM`: Recursive `LGBMRegressor` with multi-horizon lags `[1, 2, 3, 7, 14, 21, 28, 364]`, rolling means, `weight_col="sample_weight"`, and dynamic exogenous futures.
   - `Hybrid_Ridge_CatBoost`: `CatBoostRegressor` predicting clipped post-Ridge log residuals using causal residual momentum (`resid_mom7`, `resid_mom28`, `resid_dow7`, `resid_yr364`) + weather/holiday nonlinearities.
2. **Ensemble Weighting**:
   - Out-of-fold weights are learned strictly via leave-one-fold-out inverse-$\text{RMSE}^4$ across the 4 orthogonal families, then recomputed across all 12 folds for production inference.
   - Predictions are averaged **in log space** (geometric ensemble) and exponentiated back to natural vehicle counts.
3. **Conformal Interval Calibration**:
   - Computes signed out-of-sample log errors $e_i = \ln y_i - \ln \hat{y}_i$, estimates median log bias, scales by horizon dispersion $m(h) = \sqrt{(1 + 0.025(h-1)) / (1 + 0.025 \cdot 7.5)}$, and computes split-conformal finite-sample quantiles $q_{80}, q_{95}$ stratified across weekdays vs. weekends.
4. **Validation Results (`12 folds × 16d = 192 OOF days`)**:
   - Point Accuracy (`Ensemble_Weighted`): **MAE 1,071** · **RMSE 1,411** · **MAPE 5.69%** · **sMAPE 5.64%** · **MASE 0.540** · **Bias +0.93%** (vs. `SeasonalNaive364`: RMSE 1,839, MAPE 8.02%).
   - Empirical OOS Interval Coverage: **79.2%** (80% target) and **94.8%** (95% target).

---

## 5. CLI & Workflows Reference

### CLI Commands (`banff-planner`)
```bash
# Full daily production update (live APIs + 12-fold CV + conformal calibration + site build)
banff-planner --refresh-web --verbose

# Fast local rebuild reusing cached expanding CV & API files
banff-planner --skip-cv

# Backwards-compatible root wrappers
python extract_data.py
python update_predictions.py --skip-cv
```

### GitHub Actions Workflows
- `.github/workflows/update_data.yml`: Scheduled daily at `11:00 UTC` (and `workflow_dispatch`), installs `.[app,dev]`, runs `banff-planner --refresh-web`, runs `pytest -v`, and commits updated data/outputs/`site/` to `master`.
- `.github/workflows/deploy-pages.yml`: Uses `actions/upload-pages-artifact@v3` + `actions/deploy-pages@v4` on `master` pushes to `site/**`, upon successful completion of `Update Traffic Data & Forecast` (avoiding `GITHUB_TOKEN` recursive push suppression), or via manual `workflow_dispatch`.
- `.github/workflows/ci.yml`: Runs `pytest -v` (ingestion guards, statistical tests, conformal monotonicity, Node.js `BanffPlannerLogic` ↔ Python exact parity across 36 combinations, and headless Streamlit `AppTest`) on pushes and pull requests.

---

## 6. Verification Suite

```bash
.venv/bin/pytest -v
```
Tests verify:
- Tableau `.hyper` deduplication, 23h completeness filter, nullable Counter 3, 4× crosstab inflation rejection, partial trailing snapshot trimming, and StatCan `vectorId`-keyed lookup.
- Absence of any municipal Zero Waste file references across the entire package and generated site.
- Day-type conditional percentiles, strictly contiguous calendar window selection, and unbiased trend comparisons.
- Non-cloning across candidate ensemble families, entrance sum coherence (`west + east == Ensemble`), and conformal monotonicity.
- 100% exact decision-logic parity across all 36 combinations between [`analytics.py`](src/banff_planner/analytics.py) and client-side JavaScript [`BanffPlannerLogic`](src/banff_planner/site_builder.py) under Node.js `vm`.
- Lightweight dependency boundary + full multi-filter rendering of [`traffic_app.py`](traffic_app.py) via `streamlit.testing.v1.AppTest`.
