"""Central configuration for Banff Visit Planner paths, web APIs, exogenous metadata, and UI mappings."""

from pathlib import Path
from typing import Dict, Optional, Tuple

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
CACHE_DIR = DATA_DIR / "cache"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
SITE_DIR = PROJECT_ROOT / "site"

# Legacy root files maintained for Streamlit / backwards compatibility
ROOT_TRAFFIC_TSV = PROJECT_ROOT / "TW Traffic _data.csv"
ROOT_PREDICTIONS_CSV = PROJECT_ROOT / "predictions.csv"
RAW_TRAFFIC_CSV = RAW_DATA_DIR / "tw_traffic_data.csv"
RAW_WEATHER_CSV = RAW_DATA_DIR / "weather_df.csv"

# Coordinates & weather station metadata
# BANFF CS official ECCC climate station (1995-present)
ECCC_STATION_ID = "3050519"
ECCC_STATION_NAME = "BANFF CS"
STATION_LAT = 51.193358333333336
STATION_LON = -115.55223611111111

# Town of Banff centre for tourist forecast display
BANFF_LAT = 51.1784
BANFF_LON = -115.5708

# Entrance counter mapping in Tableau export
ENTRANCE_COLUMNS: Dict[str, str] = {
    "Vehicles Per Day": "Combined Town Traffic (West + East Entrances)",
    "West_Entrance": "West Access — Bow Valley Pkwy / Norquay Rd (Counter 2)",
    "East_Entrance": "East Access — Banff Ave / Trans-Canada Hwy 1 (Counter 1)",
    "Mountain_Ave": "Mountain Ave / Sulphur Mountain Corridor (Counter 3)",
}

# Verified Statistics Canada WDS Vector IDs (indexed strictly by integer vectorId)
# Excludes all Zero Waste data per project specification.
STATCAN_VECTORS: Dict[int, str] = {
    41692327: "AB_CPI_All",
    41692328: "AB_CPI_Food",
    41692401: "AB_CPI_Gasoline",
    41692458: "AB_CPI_Energy",
    41690973: "CA_CPI_All",
    42170714: "AB_FoodServices_Receipts_SA",
    2064510: "AB_LFS_Pop",
    2064512: "AB_LFS_Emp_SA",
    2064516: "AB_LFS_UnempRate_SA",
    3075059: "AB_AvgHourlyWage",
    1446859973: "AB_Retail_Total_SA",
    65201210: "CA_GDP_AllInd_M",
    65201468: "CA_GDP_AccomFood_M",
    1277197675: "AB_Visitors_US",
    1277197676: "AB_Visitors_Overseas",
}

STATCAN_VECTOR_DESCRIPTIONS: Dict[str, str] = {
    "AB_CPI_All": "Alberta CPI, All-items (2002=100)",
    "AB_CPI_Food": "Alberta CPI, Food (2002=100)",
    "AB_CPI_Gasoline": "Alberta CPI, Gasoline (2002=100)",
    "AB_CPI_Energy": "Alberta CPI, Energy (2002=100)",
    "CA_CPI_All": "Canada CPI, All-items (2002=100)",
    "AB_FoodServices_Receipts_SA": "Alberta Food Services & Drinking Places Receipts, SA ($000s)",
    "AB_LFS_Pop": "Alberta Population 15+ (000s)",
    "AB_LFS_Emp_SA": "Alberta Employment, SA (000s)",
    "AB_LFS_UnempRate_SA": "Alberta Unemployment Rate, SA (%)",
    "AB_AvgHourlyWage": "Alberta Average Hourly Wage ($ CAD)",
    "AB_Retail_Total_SA": "Alberta Retail Trade Sales, SA ($000s)",
    "CA_GDP_AllInd_M": "Canada Monthly Real GDP All Industries, SAAR",
    "CA_GDP_AccomFood_M": "Canada Monthly Real GDP Accommodation & Food Services [NAICS 72]",
    "AB_Visitors_US": "US Residents Entering Canada via Alberta (monthly count)",
    "AB_Visitors_Overseas": "Overseas Residents Entering Canada via Alberta (monthly count)",
}

# Programmatic Data Endpoints
BANFF_TABLEAU_WORKBOOK_URL = "https://public.tableau.com/workbooks/BanffTrafficData-GS.twb"
STATCAN_WDS_VECTOR_URL = "https://www150.statcan.gc.ca/t1/wds/rest/getDataFromVectorsAndLatestNPeriods"
BOC_VALET_FX_URL = "https://www.bankofcanada.ca/valet/observations/FXUSDCAD/json"
BOC_VALET_LEGACY_FX_URL = "https://www.bankofcanada.ca/valet/observations/IEXE0101/json"
ALBERTA_PARK_ATTENDANCE_URL = (
    "https://api.economicdata.alberta.ca/data"
    "?table=NationalParkAttendance&Type=Standard&Indicator=Attendance&TourismName=Banff%20National%20Park"
)
ALBERTA_HWY_COUNT_URL = "https://api.economicdata.alberta.ca/data?table=Highway_count"
ECCC_CLIMATE_DAILY_URL = "https://api.weather.gc.ca/collections/climate-daily/items"
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# Default forecasting settings
DEFAULT_HORIZON_DAYS = 16  # Ensures >= 15 full future days from any local morning run
SEASONAL_BASELINE_YEARS = 3
SEASONAL_WINDOW_DAYS = 10

# WMO Weather interpretation code mapping -> (label, icon_key, comfort_penalty)
WMO_WEATHER_MAP: Dict[int, Tuple[str, str, float]] = {
    0: ("Clear sky", "sun", 0.0),
    1: ("Mainly clear", "sun-cloud", 2.0),
    2: ("Partly cloudy", "partly-cloudy", 5.0),
    3: ("Overcast", "cloud", 10.0),
    45: ("Fog", "fog", 18.0),
    48: ("Depositing rime fog", "fog", 22.0),
    51: ("Light drizzle", "drizzle", 20.0),
    53: ("Moderate drizzle", "drizzle", 26.0),
    55: ("Dense drizzle", "rain", 32.0),
    56: ("Light freezing drizzle", "sleet", 38.0),
    57: ("Dense freezing drizzle", "sleet", 44.0),
    61: ("Slight rain", "rain", 28.0),
    63: ("Moderate rain", "rain", 38.0),
    65: ("Heavy rain", "rain-heavy", 48.0),
    66: ("Light freezing rain", "sleet", 45.0),
    67: ("Heavy freezing rain", "sleet", 55.0),
    71: ("Light snow", "snow", 22.0),
    73: ("Moderate snow", "snow", 34.0),
    75: ("Heavy snow", "snow-heavy", 48.0),
    77: ("Snow grains", "snow", 26.0),
    80: ("Slight rain showers", "rain", 25.0),
    81: ("Moderate rain showers", "rain", 36.0),
    82: ("Violent rain showers", "rain-heavy", 50.0),
    85: ("Slight snow showers", "snow", 24.0),
    86: ("Heavy snow showers", "snow-heavy", 42.0),
    95: ("Thunderstorm", "storm", 55.0),
    96: ("Thunderstorm w/ hail", "storm", 65.0),
    99: ("Severe thunderstorm w/ hail", "storm", 70.0),
}


def describe_wmo_code(code: Optional[float]) -> Tuple[str, str, float]:
    """Return (condition_label, icon_key, base_penalty) for a WMO weather code."""
    if code is None or code != code:  # NaN check
        return ("Seasonal conditions", "partly-cloudy", 10.0)
    return WMO_WEATHER_MAP.get(int(round(float(code))), ("Variable mountain weather", "partly-cloudy", 12.0))
