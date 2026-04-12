from __future__ import annotations

from pathlib import Path

import pandas as pd


HISTORICAL_RENAME_MAP = {
    "Day of Time Stamp": "Date",
    "Combined TW": "Vehicles Per Day",
}


def load_csv(path: Path, **read_kwargs) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path.name} not found in {path.parent}")
    return pd.read_csv(path, **read_kwargs)


def normalize_historical_traffic(
    df: pd.DataFrame,
    *,
    incomplete_day_ratio: float = 0.25,
    recent_window: int = 30,
) -> pd.DataFrame:
    df = df.copy()
    df.columns = [col.strip() for col in df.columns]

    missing = [src for src in HISTORICAL_RENAME_MAP if src not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in historical data: {', '.join(missing)}")

    df = df.rename(columns=HISTORICAL_RENAME_MAP)
    df = df[df["Date"] != "Grand Total"]
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Vehicles Per Day"] = pd.to_numeric(df["Vehicles Per Day"], errors="coerce")
    df = df.dropna(subset=["Date", "Vehicles Per Day"]).sort_values("Date")
    df = df[~df["Date"].duplicated(keep="first")]

    if df.empty:
        return df

    latest_date = df["Date"].max()
    latest_mask = df["Date"] == latest_date
    latest_value = df.loc[latest_mask, "Vehicles Per Day"].iloc[0]
    recent_series = df["Vehicles Per Day"].tail(recent_window)
    recent_median = recent_series.median()

    if pd.notna(recent_median) and recent_median > 0 and latest_value < recent_median * incomplete_day_ratio:
        df = df.loc[~latest_mask].copy()

    return df
