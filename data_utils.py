"""Backward-compatible shared CSV loading and validated historical traffic helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HISTORICAL_RENAME_MAP = {
    "Day of Time Stamp": "Date",
    "Combined TW": "Vehicles Per Day",
    "Tw (002)": "West_Entrance",
    "Sum of TW": "East_Entrance",
    "Tw (003)": "Mountain_Ave",
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
    """Normalize Town of Banff traffic export, guard against 4x crosstab bugs & partial latest day."""
    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]

    required_src = ["Day of Time Stamp", "Combined TW"]
    missing = [src for src in required_src if src not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in historical data: {', '.join(missing)}")

    df = df.rename(columns={k: v for k, v in HISTORICAL_RENAME_MAP.items() if k in df.columns})
    df = df[df["Date"].astype(str).str.strip() != "Grand Total"].copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    for col in ["Vehicles Per Day", "West_Entrance", "East_Entrance", "Mountain_Ave"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Date", "Vehicles Per Day"]).sort_values("Date")
    df = df[~df["Date"].duplicated(keep="first")].reset_index(drop=True)

    if df.empty:
        return df

    # 4x Tableau aggregation ratio sanity guard
    if {"West_Entrance", "East_Entrance"}.issubset(df.columns):
        denom = (df["West_Entrance"] + df["East_Entrance"]).replace(0, np.nan)
        ratio = float((df["Vehicles Per Day"] / denom).median())
        if pd.notna(ratio) and ratio > 1.5:
            raise ValueError(
                f"Traffic dataset failed Tableau aggregation sanity guard (Combined/(West+East)={ratio:.2f})."
            )

    latest_date = df["Date"].max()
    latest_mask = df["Date"] == latest_date
    latest_value = float(df.loc[latest_mask, "Vehicles Per Day"].iloc[0])
    prior = df.loc[~latest_mask, "Vehicles Per Day"].tail(recent_window)
    recent_median = float(prior.median()) if not prior.empty else float(df["Vehicles Per Day"].median())

    if pd.notna(recent_median) and recent_median > 0 and latest_value < recent_median * incomplete_day_ratio:
        df = df.loc[~latest_mask].copy()

    return df
