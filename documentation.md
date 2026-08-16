# Banff Traffic Insights & Visit Planner Documentation

An interactive Streamlit app that helps tourists decide when to visit Banff with lower expected traffic, using historical data and AI-generated forecasts.

## Features

- **Decision-First Visit Planner**: Highlights the best day to visit and the busiest day to avoid
- **Traffic Forecasts**: Shows the upcoming 15-day traffic outlook
- **Planning Picks**: Surfaces top recommended days and days to avoid
- **Trip Filters**: Restrict recommendations to weekdays only or weekends only
- **Day of Week Analysis**: Compare average traffic by day of week
- **Monthly Patterns**: Visualize seasonal trends across the three most recent years
- **Automated Data Updates**: Built-in data extraction script and GitHub Actions workflow to keep historical traffic data and AI forecasts up-to-date

## Data Source

Combined two-way traffic data from the Town of Banff, Alberta (July 2013 - Present)
- [Banff Traffic Data Dashboard](https://public.tableau.com/app/profile/banff.gis/viz/BanffTrafficData-GS/TWTraffic)

## Automated Data Updates

The historical dataset (`TW Traffic _data.csv`) and forecast data (`predictions.csv`) are kept up-to-date automatically:
- A Python script (`extract_data.py`) fetches the latest `.twb` workbook from Tableau Public, extracts the underlying `.hyper` database, and aggregates the daily traffic counts. Tableau Public rate-limits bursts, so the download retries up to five times with exponential backoff (honouring `Retry-After`) before giving up.
- Another script (`update_predictions.py`) runs the pretrained neural forecasting models to dynamically generate the next 15 days of predictions based on the latest available data.
- A GitHub Actions workflow (`.github/workflows/update_data.yml`) runs on the 1st and 16th of each month at 06:00 UTC, and can also be triggered manually via `workflow_dispatch`.
- The workflow downloads the pretrained `modelos_exo` bundle from the `modelos-exo-v1` GitHub Release before generating forecasts. The bundle is not committed to the repository.
- The workflow automatically commits and pushes the updated CSVs back to the repository.

Each run produces a 15-day forecast starting the day after the last complete day of history, so the published forecast window shifts forward with every run. The most recent day in the source data is dropped automatically when it is a partial day (see `normalize_historical_traffic` in `data_utils.py`).

## Visit Signals

- 🟢 **Best day to visit** - Lowest expected traffic in the current forecast window
- 🟩 **Good day to visit** - Lower-than-usual traffic
- 🟡 **Busy day** - Higher traffic, but still workable
- 🔴 **Avoid if possible** - Highest expected traffic

## Installation

1. Clone this repository:
```bash
git clone <your-repository-url>
cd visit_planner_app
```

2. Create and activate a virtual environment:
```bash
python3 -m venv .venv && source .venv/bin/activate
```

3. Install required dependencies:
```bash
pip install -r requirements.txt
```

If you also want to run the data refresh and forecasting scripts locally, install the update pipeline dependencies too:

```bash
pip install -r requirements-update.txt
```

4. Ensure you have the required data files:
   - `TW Traffic _data.csv` - Historical traffic data (updated automatically via GitHub Actions, or manually via `python extract_data.py`)
   - `predictions.csv` - Forecast data (updated automatically via GitHub Actions, or manually via `python update_predictions.py`)

5. To regenerate forecasts locally, download the pretrained model bundle. It is excluded from version control, so `update_predictions.py` fails without it:
```bash
gh release download modelos-exo-v1 --pattern modelos_exo.zip && unzip -q modelos_exo.zip
```

Running the app itself only needs `requirements.txt` plus the two CSVs — the model bundle is not required.

## Running the App

```bash
streamlit run traffic_app.py
```

The app will open in your default web browser at `http://localhost:8501`

## Deployment

This app can be deployed on:
- **Streamlit Community Cloud** (free)
- **Heroku**
- **AWS/GCP/Azure**
- Any platform supporting Python web apps

### Deploy to Streamlit Community Cloud

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Set the main file path to `traffic_app.py`
5. Deploy!

## Project Structure

```text
Visit_Planner_App/
├── .github/workflows/       # GitHub Actions for automated data updates
├── .devcontainer/           # Dev container / Codespaces configuration
├── traffic_app.py           # Main Streamlit application
├── data_utils.py            # Shared data loading and normalization helpers
├── extract_data.py          # Script to download and process Tableau data
├── update_predictions.py    # Script to dynamically generate AI traffic forecasts
├── TW Traffic _data.csv     # Historical traffic data (UTF-16, tab-separated)
├── predictions.csv          # Forecast data
├── requirements.txt         # Runtime dependencies for the Streamlit app
├── requirements-update.txt  # Extra dependencies for data extraction and forecast updates
├── visit_planner.png        # README header image
├── documentation.md         # Technical project documentation
├── LICENSE                  # MIT License
└── README.md                # Public repository overview
```

Not tracked in the repository (created locally or fetched on demand):

```text
├── modelos_exo/             # Pretrained neural forecast models (GitHub Release modelos-exo-v1)
├── lightning_logs/          # PyTorch Lightning training output
└── .venv/                   # Local virtual environment
```

## Technologies Used

**Python 3.13** (pinned in CI and used locally)

App runtime (`requirements.txt`):
- **Streamlit** - Web app framework
- **Pandas** - Data manipulation
- **Plotly** - Interactive visualizations
- **NumPy** - Numerical operations

Update pipeline (`requirements-update.txt`):
- **requests** - Downloads the Tableau workbook
- **pantab** - Reads the `.hyper` extract
- **neuralforecast** / **utilsforecast** - Forecast models and exogenous feature engineering
- **torch** / **pytorch-lightning** - Model execution backend

## Forecast Method

Forecasts come from a saved `NeuralForecast` bundle whose member models are averaged into a single `Ensemble` column. The bundle combines NBEATS, NBEATSx, and NHITSx variants trained with different seasonality windows:

- **NBEATS** (Neural Basis Expansion Analysis for Time Series)
- **NBEATSx** (NBEATS with exogenous variables)
- **NHITSx** (Neural Hierarchical Interpolation for Time Series with exogenous variables)

Exogenous features are generated in `update_predictions.py`: a trend term, Fourier terms at 7-, 365-, and 730-day season lengths, plus weekday, month, and year.

Before `predictions.csv` is written, the run is rejected if the forecast contains NaN values or falls implausibly far below recent history.

## License

Released under the MIT License. See [LICENSE](LICENSE) for the full text.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions or feedback, please open an issue on GitHub.
