"""Validated daily Town of Banff traffic loaders preserving access counters and integrity guards."""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from banff_planner.config import RAW_TRAFFIC_CSV, ROOT_TRAFFIC_TSV


def _read_traffic_raw(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Traffic CSV not found at {path}")
    with open(path, "rb") as bf:
        bom = bf.read(4)
    if bom.startswith((b"\xff\xfe", b"\xfe\xff")):
        return pd.read_csv(path, encoding="utf-16", sep="\t")
    # Also handle UTF-8 TSV or CSV cleanly
    try:
        df = pd.read_csv(path)
        if len(df.columns) == 1 and "\t" in str(df.columns[0]):
            return pd.read_csv(path, sep="\t")
        return df
    except Exception:
        return pd.read_csv(path, encoding="utf-16", sep="\t")


def load_daily_traffic(
    csv_path: Optional[Path] = None,
    *,
    incomplete_day_ratio: float = 0.25,
    recent_window: int = 30,
    start_date: str = "2013-07-01",
    reindex_daily: bool = True,
) -> pd.DataFrame:
    """Load validated daily Town of Banff two-way traffic counters.

    Preserves entrance-level breakdown:
      - 'Vehicles Per Day' (= Combined TW = West_Entrance + East_Entrance)
      - 'West_Entrance' (Tw (002) = Counter 2)
      - 'East_Entrance' (Sum of TW = Counter 1)
      - 'Mountain_Ave' (Tw (003) = Counter 3, nullable when sensor is offline)

    Enforces:
      - Grand Total row removal & deduplication by Date
      - 4x Tableau crosstab aggregation sanity guard (Combined / (West + East) <= 1.5)
      - Partial trailing snapshot day exclusion (< incomplete_day_ratio * recent 30d median)
      - Optional continuous daily reindex with short gap interpolation metadata flag
    """
    if csv_path is not None:
        path = Path(csv_path)
    elif RAW_TRAFFIC_CSV.exists():
        path = RAW_TRAFFIC_CSV
    else:
        path = ROOT_TRAFFIC_TSV

    raw = _read_traffic_raw(path)
    df = raw.copy()
    df.columns = [str(c).strip() for c in df.columns]

    date_col = "Day of Time Stamp" if "Day of Time Stamp" in df.columns else "Date"
    if date_col not in df.columns:
        raise KeyError(f"Missing date column ('Day of Time Stamp' or 'Date') in {path.name}")

    df = df[df[date_col].astype(str).str.strip() != "Grand Total"].copy()
    df["Date"] = pd.to_datetime(df[date_col].astype(str).str.strip(), errors="coerce").dt.normalize()
    df = df.dropna(subset=["Date"]).copy()

    rename_map = {
        "Combined TW": "Vehicles Per Day",
        "Combined_TW": "Vehicles Per Day",
        "Total Vehicles": "Vehicles Per Day",
        "Tw (002)": "West_Entrance",
        "Sum of TW": "East_Entrance",
        "Tw (003)": "Mountain_Ave",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    if "Vehicles Per Day" not in df.columns:
        raise KeyError(f"Missing 'Combined TW' / 'Vehicles Per Day' column in {path.name}")

    for col in ["Vehicles Per Day", "West_Entrance", "East_Entrance", "Mountain_Ave"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Vehicles Per Day"]).copy()
    df = df[df["Vehicles Per Day"] > 0].copy()
    df = df[df["Date"] >= pd.Timestamp(start_date)].copy()
    df = df.drop_duplicates(subset=["Date"], keep="first").sort_values("Date").reset_index(drop=True)

    if df.empty:
        return df

    # Sanity guard against 4x Tableau double-counting bug (e.g. 2013-07-01 = 82,668 vs 20,667)
    if {"West_Entrance", "East_Entrance"}.issubset(df.columns):
        denom = (df["West_Entrance"] + df["East_Entrance"]).replace(0, np.nan)
        ratio = float((df["Vehicles Per Day"] / denom).median())
        if pd.notna(ratio) and ratio > 1.5:
            raise ValueError(
                f"Traffic file {path.name} appears afflicted by Tableau crosstab aggregation inflation "
                f"(Total/(West+East) median={ratio:.2f})."
            )

    # Guard against partial latest snapshot day (e.g., Oct 31 = 87 vehicles vs ~12k median)
    if incomplete_day_ratio > 0 and len(df) >= 7:
        latest_idx = df["Date"].idxmax()
        latest_val = float(df.loc[latest_idx, "Vehicles Per Day"])
        hist_prior = df.loc[df.index != latest_idx, "Vehicles Per Day"].tail(recent_window)
        recent_median = float(hist_prior.median()) if not hist_prior.empty else float(df["Vehicles Per Day"].median())
        if pd.notna(recent_median) and recent_median > 0 and latest_val < recent_median * incomplete_day_ratio:
            df = df.loc[df.index != latest_idx].copy()

    df = df.sort_values("Date").reset_index(drop=True)

    # Ensure West_Entrance / East_Entrance exist even on minimal synthetic test frames
    if "West_Entrance" not in df.columns:
        df["West_Entrance"] = np.nan
    if "East_Entrance" not in df.columns:
        df["East_Entrance"] = np.nan
    if "Mountain_Ave" not in df.columns:
        df["Mountain_Ave"] = np.nan

    # Mark exact observed rows vs rare missing calendar days
    df["is_observed"] = True
    if reindex_daily and len(df) > 1:
        full_days = pd.date_range(df["Date"].min(), df["Date"].max(), freq="D")
        df_idx = df.set_index("Date").reindex(full_days)
        df_idx.index.name = "Date"
        df_idx["is_observed"] = df_idx["is_observed"].fillna(False).astype(bool)
        # Fill occasional 1-2 day counter outages cleanly for strictly continuous daily lag indexing,
        # while retaining exact observed flag so training/metrics track genuine observations
        for col in ["Vehicles Per Day", "West_Entrance", "East_Entrance"]:
            if df_idx[col].isna().any() and df_idx[col].notna().sum() >= 14:
                same_dow_med = (
                    df_idx[col]
                    .groupby(df_idx.index.day_of_week)
                    .transform(lambda s: s.interpolate(method="linear", limit_direction="both"))
                )
                df_idx[col] = df_idx[col].fillna(same_dow_med)
        df = df_idx.reset_index()

    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Month Name"] = df["Date"].dt.strftime("%b")
    df["Day of Week"] = df["Date"].dt.day_name()
    df["Day of Week Num"] = df["Date"].dt.day_of_week
    df["Is Weekend"] = df["Day of Week Num"].isin([5, 6])
    return df
