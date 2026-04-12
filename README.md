# Banff Traffic Insights & Visit Planner

An interactive Streamlit app that helps tourists decide when to visit Banff with lower expected traffic, using historical data and AI-generated forecasts.

## Features

- **Decision-First Visit Planner**: Highlights the best day to visit and the busiest day to avoid
- **Traffic Forecasts**: Shows the upcoming 15-day traffic outlook
- **Planning Picks**: Surfaces top recommended days and days to avoid
- **Day of Week Analysis**: Compare average traffic by day of week
- **Monthly Patterns**: Visualize seasonal trends across multiple years
- **Automated Data Updates**: Built-in data extraction script and GitHub Actions workflow to keep historical traffic data and AI forecasts up-to-date

## Data Source

Combined two-way traffic data from the Town of Banff, Alberta (July 2013 - Present)
- [Banff Traffic Data Dashboard](https://public.tableau.com/app/profile/banff.gis/viz/BanffTrafficData-GS/TWTraffic)

## Automated Data Updates

The historical dataset (`TW Traffic _data.csv`) and forecast data (`predictions.csv`) are kept up-to-date automatically:
- A Python script (`extract_data.py`) fetches the latest `.twb` workbook from Tableau Public, extracts the underlying `.hyper` database, and aggregates the daily traffic counts.
- Another script (`update_predictions.py`) runs the pretrained neural forecasting models to dynamically generate the next 15 days of predictions based on the latest available data.
- A GitHub Actions workflow (`.github/workflows/update_data.yml`) checks daily and only runs the update pipeline every 15 days starting from April 15, 2026.
- The workflow automatically commits and pushes the updated CSVs back to the repository.

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

2. Activate the virtual environment:
```bash
source .venv/bin/activate
```

3. Install required dependencies:
```bash
pip install -r requirements.txt
```

4. Ensure you have the required data files:
   - `TW Traffic _data.csv` - Historical traffic data (updated automatically via GitHub Actions, or manually via `python extract_data.py`)
   - `predictions.csv` - Forecast data (updated automatically via GitHub Actions, or manually via `python update_predictions.py`)

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

```
Visit_Planner_App/
├── .github/workflows/       # GitHub Actions for automated data updates
├── traffic_app.py           # Main Streamlit application
├── data_utils.py            # Shared data loading and normalization helpers
├── extract_data.py          # Script to download and process Tableau data
├── update_predictions.py    # Script to dynamically generate AI traffic forecasts
├── modelos_exo/             # Pretrained neural forecast models
├── TW Traffic _data.csv     # Historical traffic data
├── predictions.csv          # Forecast data
├── requirements.txt         # Python dependencies
├── .gitignore               # Git ignore file
└── README.md                # This file
```

## Technologies Used

- **Python 3.8+**
- **Streamlit** - Web app framework
- **Pandas** - Data manipulation
- **Plotly** - Interactive visualizations
- **NumPy** - Numerical operations

## Forecast Method

The forecasts are generated using neural ensemble forecasting combining:
- **NBEATSx** (Neural Basis Expansion Analysis for Time Series with exogenous variables)
- **NHITSx** (Neural Hierarchical Interpolation for Time Series with exogenous variables)

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions or feedback, please open an issue on GitHub.
