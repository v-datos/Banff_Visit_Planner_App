<p align="center">
  <img src="visit_planner.png" alt="Banff Visit Planner Header" width="800">
</p>

# Banff Visit Planner

[![CI Tests & Site Parity](https://github.com/v-datos/Banff_Visit_Planner_App/actions/workflows/ci.yml/badge.svg)](https://github.com/v-datos/Banff_Visit_Planner_App/actions/workflows/ci.yml)

- **Primary Fast Web App (GitHub Pages):** [v-datos.github.io/Banff_Visit_Planner_App](https://v-datos.github.io/Banff_Visit_Planner_App/)
- **Streamlit Companion App:** [banff-visit-planner-app.streamlit.app](https://banff-visit-planner-app.streamlit.app/)

**Banff Visit Planner** helps travelers pick the best upcoming days to visit Banff National Park over a rolling **16-day forecast horizon**, balancing **lower road congestion** with **comfortable mountain weather** and calibrated **conformal uncertainty intervals**.

---

## Key Capabilities (v2.0)

- **Instant Trip Decision Cards:** Highlights the single **Recommended Day**, the **Busiest Day to Avoid**, and the **Best 1–4 Day Strictly Contiguous Stay** matching your filter (`All days`, `Weekdays`, or `Weekends`) and priority (`Balanced`, `Least Traffic`, or `Best Weather`).
- **16-Day Weather + Traffic Integration:** Surfaces Banff highs/lows, precipitation probability, rainfall/snowfall, and a 4-season **Weather Comfort Score** alongside daily expected vehicle entrances.
- **Fair Dual Crowd Percentiles:** Evaluates every day both **against the same day-of-week** (e.g., comparing upcoming Saturdays strictly against recent seasonal Saturdays) and **overall within the ±10-day seasonal window**.
- **Conformal Prediction Intervals (80% & 95%):** Log-scale split-conformal bounds stratified by weekday/weekend volatility and horizon step—empirically validated at **79.2% (80% target)** and **94.8% (95% target)** coverage across 192 unseen out-of-sample days.
- **Transparent Day Deep Dive:** Shows causal **driver attributions** (`Why this forecast?`), entrance breakdown (**West / Norquay–Bow Valley** vs **East / Banff Ave–Hwy 1** & **Mountain Ave**), and practical transit/arrival tips.
- **365-Day Year-Round Seasonal Guide & Accuracy Audit:** Interactive 365-day crowd/temperature heatmap, day-of-week & monthly seasonality curves, full 12-fold CV benchmark table, and live data freshness status.

---

## Multi-Family Forecasting Pipeline & Exogenous Sources

All models train strictly in **multiplicative log space ($\ln y$)** on deduplicated, completeness-verified (`>= 23h/day`) Town of Banff counter histories (2013–present), combined dynamically via out-of-sample **inverse-$\text{RMSE}^4$** weights:

| Model / Benchmark | MAE | RMSE | MAPE | MASE | Bias | Weight |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `Hybrid_Ridge_CatBoost` | 1,025 | 1,358 | 5.56% | 0.516 | +1.26% | 50.1% |
| **`Ensemble_Weighted` (Log Ensemble)** | **1,071** | **1,411** | **5.69%** | **0.540** | **+0.93%** | **100%** |
| `MLForecast_LGBM` | 1,256 | 1,601 | 6.81% | 0.633 | +0.93% | 25.9% |
| `Ridge_AR_Exo` | 1,396 | 1,754 | 7.34% | 0.703 | +1.80% | 18.0% |
| `SeasonalNaive364` *(baseline)* | 1,430 | 1,839 | 8.02% | 0.721 | -1.87% | — |
| `LogMSTL_ETS` | 1,924 | 2,308 | 9.38% | 0.969 | -2.01% | 6.0% |

*Evaluated over 12 multi-season expanding-window folds × 16-day horizons (192 unseen out-of-sample days).*

### Programmatic Data Feeds (Zero Waste Strictly Excluded)
1. **Town of Banff Traffic Counters** — Open Tableau Public `.twb` → `.hyper` extraction (`West_Entrance`, `East_Entrance`, `Mountain_Ave`) with 4× inflation and partial-day guards.
2. **Banff Climate & 16-Day Weather NWP** — ECCC official station `BANFF CS (3050519)` + Open-Meteo ERA5 bias-calibrated archive + 16-day NWP forecast.
3. **Bank of Canada FX** — Daily `FXUSDCAD` (spliced with pre-2017 `IEXE0101`).
4. **Statistics Canada WDS** — Alberta CPI (Gasoline, Food, Energy, All-Items), Labour Force Unemployment/Wage/Employment, Food Services, Real GDP, and International Travellers.
5. **Alberta Economic Dashboard** — Hwy 1 Banff Park Gates counts & Banff National Park Attendance.
6. **Alberta Holidays & Bridges** — Statutory holidays (`holidays.CA(subdiv="AB")`), Easter Monday, long-weekend Fridays/Sundays/Mondays, bridge days, Larch Season, and Christmas break.

Municipal Zero Waste series are intentionally **excluded** from all pipelines, features, models, and artifacts.

---

## Repository Structure

```text
Banff_Visit_Planner_App/
├── .github/workflows/
│   ├── update_data.yml          # Daily (11:00 UTC) refresh, CV, conformal calibration, pytest gate & site build
│   ├── deploy-pages.yml         # Publishes site/ cleanly to GitHub Pages
│   └── ci.yml                   # Python + Node.js test & JS/Python logic parity suite
├── src/banff_planner/
│   ├── config.py                # Endpoints, StatCan vectors, WMO weather map & repo paths
│   ├── data_loader.py           # Validated traffic CSV/TSV loader & 4x/partial-day guards
│   ├── exogenous.py             # Live Tableau .hyper, ECCC, Open-Meteo, BoC, StatCan & Alberta Economic ETL
│   ├── features.py              # Alberta holidays/bridges, leap-safe harmonics, 4-season weather comfort & causal lags
│   ├── models.py                # Log-scale orthogonal model suite, entrance split & causal attributions
│   ├── evaluation.py            # 12-fold multi-season expanding CV & split-conformal intervals (80% / 95%)
│   ├── analytics.py             # Dual percentiles, strictly contiguous best windows & unbiased trend scoring
│   ├── pipeline.py              # End-to-end orchestrator writing outputs/*, predictions.csv & site/*
│   ├── site_builder.py          # Zero-dependency interactive static HTML5/Canvas app builder
│   └── cli.py                   # `banff-planner` CLI entry point
├── tests/                       # Ingestion, mathematical analytics, conformal calibration & JS-parity test suite
├── site/                        # Prebuilt self-contained static web app for GitHub Pages
├── outputs/                     # Forecast JSON payload, enriched CSV & CV benchmark reports
├── data/                        # Raw normalized extracts & API caches
├── traffic_app.py               # Lightweight Streamlit companion UI (reads requirements.txt only)
├── update_predictions.py        # CLI wrapper invoking `banff_planner.cli:main`
├── extract_data.py              # CLI wrapper refreshing Town of Banff Tableau counts
├── data_utils.py                # Shared lightweight loading wrappers
├── predictions.csv              # Root-compatible published forecast CSV
├── TW Traffic _data.csv         # Daily historical traffic dataset (UTF-16 TSV)
├── pyproject.toml               # Package metadata & dependencies
├── requirements.txt             # Minimal runtime dependencies for Streamlit Community Cloud
├── requirements-update.txt      # Editable pipeline + test dependencies (`-e .[app,dev]`)
└── documentation.md             # Full technical & mathematical architecture guide
```

---

## Quick Start

### 1. View or Run Locally
Open [`site/index.html`](site/index.html) directly in any browser—no server required—or run the Streamlit companion locally:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run traffic_app.py
```

### 2. Re-run the Full Forecasting Pipeline & Rebuild Site
```bash
pip install -e ".[app,dev]"
banff-planner --refresh-web --verbose   # Refresh APIs, run 12-fold CV, calibrate conformal PIs, build site/
pytest -v                               # Verify mathematical invariants & JS/Python exact parity
```

For faster local iteration reusing cached expanding CV & API snapshots:
```bash
banff-planner --skip-cv
```

## Enabling GitHub Pages (One-Time Repository Setting)
Once merged to `master`, ensure **Settings → Pages → Build and deployment → Source** is set to **GitHub Actions**. From that point onward, `.github/workflows/deploy-pages.yml` deploys automatically after every `master` update and every daily run of `Update Traffic Data & Forecast`.
