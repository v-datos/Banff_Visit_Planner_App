"""Unit and integrity tests for Phase 1 traffic ingestion and exogenous data normalization."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from banff_planner.config import PROJECT_ROOT
from banff_planner.data_loader import load_daily_traffic
from banff_planner.exogenous import _extract_daily_traffic_from_hyper_frames, fetch_statcan_vectors


def _make_hourly_counter(counter: str, date_str: str, tw_per_hour: float, hours: int = 24) -> pd.DataFrame:
    ts = pd.date_range(f"{date_str} 00:00:00", periods=hours, freq="h")
    return pd.DataFrame({"Counter": [counter] * hours, "TimeStamp": ts, "TW": [tw_per_hour] * hours})


def test_tableau_hyper_dedup_completeness_and_nullable_counter3():
    """Verify (Counter, TimeStamp) dedup, >=23h/day completeness, and nullable Counter 3."""
    day_full = "2026-11-10"
    day_partial = "2026-11-11"

    c1_full = _make_hourly_counter("1", day_full, 250.0, hours=24)  # East = 6000
    c2_full = _make_hourly_counter("2", day_full, 300.0, hours=24)  # West = 7200
    c3_full = _make_hourly_counter("3", day_full, 100.0, hours=24)  # Mtn Ave = 2400

    # Duplicate table subset overlapping day_full (must NOT double count!)
    dup_overlap = _make_hourly_counter("1", day_full, 250.0, hours=12)

    # Partial trailing snapshot day (only 3 hours) must be dropped automatically
    c1_part = _make_hourly_counter("1", day_partial, 250.0, hours=3)
    c2_part = _make_hourly_counter("2", day_partial, 300.0, hours=3)

    # Offline Counter 3 on a second complete day -> must remain <NA>, NOT 0
    day_no_c3 = "2026-11-09"
    c1_noc3 = _make_hourly_counter("1", day_no_c3, 200.0, hours=24)
    c2_noc3 = _make_hourly_counter("2", day_no_c3, 250.0, hours=24)

    frames = {
        "Table1": pd.concat([c1_full, c2_full, c3_full, c1_part, c2_part, c1_noc3, c2_noc3], ignore_index=True),
        "Table2_Dup": dup_overlap,
    }
    daily = _extract_daily_traffic_from_hyper_frames(frames, min_hours_per_day=23, start_date="2026-11-01")

    assert len(daily) == 2
    assert list(daily["Day of Time Stamp"]) == ["November 9, 2026", "November 10, 2026"]
    # November 9 has no Counter 3 -> strictly NA
    assert pd.isna(daily.loc[0, "Tw (003)"])
    assert int(daily.loc[0, "Combined TW"]) == 200 * 24 + 250 * 24
    # November 10 deduplicated -> exact 24h sums
    assert int(daily.loc[1, "Sum of TW"]) == 6000
    assert int(daily.loc[1, "Tw (002)"]) == 7200
    assert int(daily.loc[1, "Tw (003)"]) == 2400
    assert int(daily.loc[1, "Combined TW"]) == 13200


def test_load_daily_traffic_guards_4x_inflation_and_partial_tail(tmp_path: Path):
    """Verify 4x Tableau inflation raises ValueError and partial trailing day is trimmed."""
    bad_path = tmp_path / "bad_4x.csv"
    pd.DataFrame(
        {
            "Day of Time Stamp": ["July 1, 2025", "July 2, 2025", "Grand Total"],
            "Combined TW": [80000, 84000, 164000],
            "Tw (002)": [10000, 10500, 20500],
            "Sum of TW": [10000, 10500, 20500],
        }
    ).to_csv(bad_path, index=False, sep="\t", encoding="utf-16")

    with pytest.raises(ValueError, match="4x|inflation"):
        load_daily_traffic(bad_path)

    # Now test valid history + partial trailing day + missing calendar day reindexing
    dates = pd.date_range("2026-10-01", "2026-10-20", freq="D")
    rows = []
    for i, d in enumerate(dates):
        if i == 10:
            # Skip Oct 11 to test gap filling + is_observed flag
            continue
        vol = 120 if i == len(dates) - 1 else 14000 + (i % 7) * 400
        rows.append(
            {
                "Day of Time Stamp": f"{d.strftime('%B')} {d.day}, {d.year}",
                "Combined TW": vol,
                "Tw (002)": int(vol * 0.55),
                "Sum of TW": vol - int(vol * 0.55),
            }
        )
    ok_path = tmp_path / "valid_with_partial.csv"
    pd.DataFrame(rows).to_csv(ok_path, index=False, sep="\t", encoding="utf-16")

    loaded = load_daily_traffic(ok_path, incomplete_day_ratio=0.25, reindex_daily=True)
    # Trailing partial Oct 20 removed -> max date Oct 19
    assert loaded["Date"].max() == pd.Timestamp("2026-10-19")
    # Oct 11 imputed continuously and flagged as is_observed=False
    gap_row = loaded.loc[loaded["Date"] == pd.Timestamp("2026-10-11")].iloc[0]
    assert bool(gap_row["is_observed"]) is False
    assert gap_row["Vehicles Per Day"] > 10000


def test_statcan_vectors_keyed_by_vector_id_even_if_out_of_order(tmp_path: Path):
    """Ensure StatCan response parsing maps each item strictly by integer vectorId, never list position."""

    class _FakeResp:
        def raise_for_status(self):
            return None

        def json(self):
            # Intentionally return only two out-of-order vectorIds (Gasoline CPI before All CPI)
            return [
                {
                    "status": "SUCCESS",
                    "object": {
                        "vectorId": 41692401,
                        "vectorDataPoint": [{"refPer": "2026-08-01", "value": 180.0}],
                    },
                },
                {
                    "status": "SUCCESS",
                    "object": {
                        "vectorId": 41692327,
                        "vectorDataPoint": [{"refPer": "2026-08-01", "value": 160.0}],
                    },
                },
                {
                    "status": "SUCCESS",
                    "object": {
                        "vectorId": 2064516,
                        "vectorDataPoint": [{"refPer": "2026-08-01", "value": 7.1}],
                    },
                },
            ]

    with patch("banff_planner.exogenous.requests.Session.post", return_value=_FakeResp()):
        df = fetch_statcan_vectors(latest_n=2, cache_dir=tmp_path, refresh=True)

    assert pytest.approx(float(df["AB_CPI_Gasoline"].iloc[0])) == 180.0
    assert pytest.approx(float(df["AB_CPI_All"].iloc[0])) == 160.0
    assert pytest.approx(float(df["AB_LFS_UnempRate_SA"].iloc[0])) == 7.1
    assert pytest.approx(float(df["AB_Gas_Real"].iloc[0])) == 180.0 / 160.0 * 100.0


def test_no_zero_waste_data_anywhere():
    """Confirm Zero Waste datasets and references are completely excluded per project mandate."""
    for py_file in (PROJECT_ROOT / "src" / "banff_planner").glob("*.py"):
        txt = py_file.read_text(encoding="utf-8")
        # Allow explanatory docstrings stating exclusion, forbid any CSV or feature loading
        assert "zero_waste_dashboard_data" not in txt.lower()
        assert "zerowaste" not in txt.lower()
    assert not (PROJECT_ROOT / "data" / "raw" / "zero_waste_dashboard_data.csv").exists()
