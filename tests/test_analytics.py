"""Unit tests proving mathematical fixes to tourist decision analytics in analytics.py."""

import numpy as np
import pandas as pd

from banff_planner.analytics import (
    compute_best_window,
    compute_recommendations,
    daytype_percentile,
    enrich_forecast_analytics,
    get_conformal_confidence_summary,
    get_unbiased_trend_message,
    seasonal_percentile,
)


def _make_synthetic_history_and_forecast():
    dates = pd.date_range("2023-01-01", "2026-10-31", freq="D")
    rng = np.random.default_rng(42)
    is_wknd = dates.day_of_week >= 5
    # Strong clear weekday vs weekend split (~12k weekday, ~18k weekend) + mild summer curve
    doy = dates.day_of_year
    seasonal = 4000.0 * np.sin(2.0 * np.pi * (doy - 110) / 365.25)
    base = 12000.0 + seasonal + np.where(is_wknd, 5800.0, 0.0) + rng.normal(0.0, 350.0, len(dates))
    hist = pd.DataFrame(
        {
            "Date": dates,
            "Vehicles Per Day": np.clip(base, 5000.0, None),
            "Year": dates.year,
            "Month": dates.month,
            "Day of Week": dates.day_name(),
            "Day of Week Num": dates.day_of_week,
            "Is Weekend": is_wknd,
        }
    )

    # Upcoming 15 days starting Sat Nov 7, 2026 through Sat Nov 21, 2026
    fut_dates = pd.date_range("2026-11-07", periods=15, freq="D")
    # Make weekend 1 (Nov 7-8) unusually quiet for a weekend (~13,500) and weekend 2 (Nov 14-15) very busy (~21,500)
    fut_vols = []
    for d in fut_dates:
        if d in (pd.Timestamp("2026-11-07"), pd.Timestamp("2026-11-08")):
            fut_vols.append(13500.0)
        elif d in (pd.Timestamp("2026-11-14"), pd.Timestamp("2026-11-15")):
            fut_vols.append(21500.0)
        elif d.day_of_week >= 5:
            fut_vols.append(16500.0)
        else:
            fut_vols.append(10500.0)

    fc = pd.DataFrame(
        {
            "ds": fut_dates,
            "Ensemble": fut_vols,
            "pi80_low": np.array(fut_vols) * 0.92,
            "pi80_high": np.array(fut_vols) * 1.08,
            "pi95_low": np.array(fut_vols) * 0.87,
            "pi95_high": np.array(fut_vols) * 1.13,
        }
    )
    return hist, fc


def test_daytype_percentile_eliminates_weekend_vs_weekday_confounding():
    """A quiet weekend (13,500) must score calm on daytype_percentile even when overall_percentile looks higher."""
    hist, _ = _make_synthetic_history_and_forecast()
    # In early November (~doy 311), seasonal -3k -> weekday ~10,500, weekend ~16,300
    quiet_sat = pd.Timestamp("2026-11-07")
    vol_quiet_weekend = 13500.0
    o_pct = seasonal_percentile(vol_quiet_weekend, quiet_sat, hist)
    d_pct = daytype_percentile(vol_quiet_weekend, quiet_sat, hist)

    # Compared against mostly weekdays (~71% weekdays), 13.5k sits >70th percentile overall,
    # whereas strictly vs seasonal weekends it is genuinely calm (< 25th percentile)!
    assert o_pct > 65.0
    assert d_pct < 25.0


def test_compute_best_window_never_bridges_nonconsecutive_weekends():
    """Fixes Bug #6: Weekends-only 3-day request must return contiguous 2-day Sat-Sun, never an 8-day jump."""
    hist, fc = _make_synthetic_history_and_forecast()
    enriched = enrich_forecast_analytics(fc, hist)
    wknds = enriched[enriched["Is Weekend"]].copy()

    win = compute_best_window(wknds, day_filter="Weekends only", trip_days=3, score_col="visit_score_scoped_balanced")
    start_ts = pd.Timestamp(win["start"])
    end_ts = pd.Timestamp(win["end"])

    # Strictly contiguous Sat Nov 7 -> Sun Nov 8 (gap == 1 day, actual_days == 2)
    assert win["actual_days"] == 2
    assert (end_ts - start_ts).days == 1
    assert start_ts == pd.Timestamp("2026-11-07")
    assert end_ts == pd.Timestamp("2026-11-08")

    # Also verify general 3-day contiguous window actually spans exact 2 calendar days difference
    win_all = compute_best_window(enriched, day_filter="All upcoming days", trip_days=3, score_col="visit_score_balanced")
    assert win_all["actual_days"] == 3
    assert (pd.Timestamp(win_all["end"]) - pd.Timestamp(win_all["start"])).days == 2


def test_unbiased_trend_message_respects_day_filter_and_conformal_summary():
    """Filtering to Weekends only must NOT falsely report a +50% 'Busier trend' due to weekday comparison."""
    hist, fc = _make_synthetic_history_and_forecast()
    enriched = enrich_forecast_analytics(fc, hist)
    wknds = enriched[enriched["Is Weekend"]].copy()

    lbl, detail = get_unbiased_trend_message(hist, wknds, day_filter="Weekends only")
    assert "recent weekends" in detail
    # Quiet Nov 7-8 (~13.5k) vs recent Oct weekends (~17-18k) should be quieter/steady, NEVER "Busier trend"
    assert lbl != "Busier trend"

    c_lbl, c_det = get_conformal_confidence_summary(enriched, {"empirical_cov80_pct": 79.7, "n_oof_days": 192})
    assert "conformal uncertainty" in c_det.lower()
    assert c_lbl in {"High Precision", "Standard", "Wider Spread"}

    rec = compute_recommendations(enriched, hist, day_filter="Weekends only", priority="balanced", trip_days=3)
    assert rec is not None
    assert rec.best_day["ds"] == pd.Timestamp("2026-11-07")
    assert rec.worst_day["ds"] == pd.Timestamp("2026-11-14") or rec.worst_day["ds"] == pd.Timestamp("2026-11-15")
