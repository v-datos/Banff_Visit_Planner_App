"""End-to-end validation of published artifacts, Node.js <-> Python parity, and Streamlit app smoke checks."""

from __future__ import annotations

import ast
import json
import re
import shutil
import subprocess

import numpy as np
import pandas as pd
import pytest

from banff_planner.analytics import compute_recommendations, enrich_forecast_analytics
from banff_planner.config import OUTPUTS_DIR, PROJECT_ROOT, ROOT_PREDICTIONS_CSV, SITE_DIR
from banff_planner.data_loader import load_daily_traffic


def _extract_site_payload_and_logic():
    html_path = SITE_DIR / "index.html"
    assert html_path.exists(), f"Missing generated static site at {html_path}"
    html = html_path.read_text(encoding="utf-8")

    payload_match = re.search(
        r'<script id="banff-payload-data" type="application/json">(.*?)</script>',
        html,
        flags=re.DOTALL,
    )
    logic_match = re.search(
        r'<script id="banff-planner-logic">(.*?)</script>',
        html,
        flags=re.DOTALL,
    )
    assert payload_match is not None, "Missing embedded JSON payload tag in site/index.html"
    assert logic_match is not None, "Missing banff-planner-logic module script in site/index.html"

    # Reverse HTML5 safe </ escaping
    payload = json.loads(payload_match.group(1).replace("<\\/", "</"))
    logic_js = logic_match.group(1)
    return html, payload, logic_js


def test_published_forecast_and_artifacts_integrity():
    """Verify predictions.csv, outputs/forecast_payload.json, CV benchmarks, and site/ outputs are complete and valid."""
    payload_file = OUTPUTS_DIR / "forecast_payload.json"
    site_payload_file = SITE_DIR / "data" / "forecast_payload.json"
    assert payload_file.exists()
    assert site_payload_file.exists()
    assert (SITE_DIR / ".nojekyll").exists()
    favicon_path = SITE_DIR / "favicon.svg"
    assert favicon_path.exists(), f"Missing favicon.svg at {favicon_path}"
    assert "<svg" in favicon_path.read_text(encoding="utf-8")
    index_html = (SITE_DIR / "index.html").read_text(encoding="utf-8")
    assert '<link rel="icon" type="image/svg+xml" href="./favicon.svg" />' in index_html
    assert 'href="/favicon.svg"' not in index_html
    assert ROOT_PREDICTIONS_CSV.exists()

    payload = json.loads(payload_file.read_text(encoding="utf-8"))
    fc_days = payload["forecast_days"]
    upcoming = [d for d in fc_days if not d.get("is_bridge_gap", False)]
    assert len(upcoming) >= 15, f"Expected >= 15 future days, found {len(upcoming)}"
    assert payload["data_freshness"]["zero_waste_excluded"] is True

    # Verify predictions.csv numerical sanity & monotonicity
    pred_csv = pd.read_csv(ROOT_PREDICTIONS_CSV)
    assert len(pred_csv) >= 15
    for col in ("Ensemble", "pi80_low", "pi80_high", "pi95_low", "pi95_high", "west_entrance", "east_entrance"):
        assert col in pred_csv.columns
        assert pred_csv[col].isna().sum() == 0

    assert (pred_csv["pi95_low"] <= pred_csv["pi80_low"]).all()
    assert (pred_csv["pi80_low"] <= pred_csv["Ensemble"]).all()
    assert (pred_csv["Ensemble"] <= pred_csv["pi80_high"]).all()
    assert (pred_csv["pi80_high"] <= pred_csv["pi95_high"]).all()
    assert ((pred_csv["west_entrance"] + pred_csv["east_entrance"]) - pred_csv["Ensemble"].round()).abs().max() <= 2.0

    # Ensure published diverse ensemble members differ meaningfully (zero clones)
    cands = ["LogMSTL_ETS", "Ridge_AR_Exo", "MLForecast_LGBM", "Hybrid_Ridge_CatBoost"]
    for i in range(len(cands)):
        for j in range(i + 1, len(cands)):
            diff = float(np.max(np.abs(pred_csv[cands[i]].to_numpy() - pred_csv[cands[j]].to_numpy())))
            assert diff > 10.0

    # Verify CV benchmark outperforms naive baseline and achieves sound conformal coverage
    cv = payload["cv_benchmark"]
    metrics_by_model = {m["model"]: m for m in cv["summary_metrics"]}
    assert metrics_by_model["Ensemble_Weighted"]["RMSE"] < metrics_by_model["SeasonalNaive364"]["RMSE"]
    assert 70.0 <= float(cv["empirical_cov80_pct"]) <= 92.0
    assert 88.0 <= float(cv["empirical_cov95_pct"]) <= 99.5

    # Annual planner calendar should contain all 365 non-leap days
    assert len(payload["annual_planner"]["daily_profile_365"]) == 365


def test_js_and_python_decision_logic_exact_parity():
    """Execute site/index.html JS engine under Node.js and assert exact parity with Python analytics.py."""
    node_bin = shutil.which("node")
    if node_bin is None:
        pytest.skip("Node.js not available in PATH")

    html, payload, logic_js = _extract_site_payload_and_logic()
    assert "zero_waste_dashboard_data" not in html.lower()

    # Check syntax of all inline script tags cleanly via Node vm
    node_runner = """
const vm = require('vm');
const fs = require('fs');
const input = JSON.parse(fs.readFileSync(0, 'utf8'));
const ctx = { console, Date, Math, Number, String, Object, Array };
ctx.globalThis = ctx;
vm.runInNewContext(input.logicJs, ctx, { filename: 'banff-planner-logic.js' });
const L = ctx.BanffPlannerLogic;
if (!L || typeof L.computeRecommendations !== 'function') {
  throw new Error('BanffPlannerLogic missing computeRecommendations');
}
const out = {};
for (const df of input.dayFilters) {
  for (const pr of input.priorities) {
    for (const td of input.tripDaysList) {
      const rec = L.computeRecommendations(input.payload, df, pr, td);
      out[df + '|' + pr + '|' + td] = rec ? {
        bestDs: rec.bestDay.ds,
        worstDs: rec.worstDay.ds,
        winStart: rec.bestWindow.start,
        winEnd: rec.bestWindow.end,
        winDays: rec.bestWindow.actual_days,
        trendLabel: rec.trend.label,
        busyCount: rec.busyCount
      } : null;
    }
  }
}
process.stdout.write(JSON.stringify(out));
"""
    day_filters = ["All upcoming days", "Weekdays only", "Weekends only"]
    priorities = ["balanced", "quiet", "weather"]
    trip_days_list = [1, 2, 3, 4]

    proc = subprocess.run(
        [node_bin, "-e", node_runner],
        input=json.dumps(
            {
                "logicJs": logic_js,
                "payload": payload,
                "dayFilters": day_filters,
                "priorities": priorities,
                "tripDaysList": trip_days_list,
            }
        ),
        capture_output=True,
        text=True,
        check=True,
    )
    js_results = json.loads(proc.stdout)

    # Now compute identical combinations in Python using recent_history + upcoming forecast_days
    hist_recent = pd.DataFrame(payload["recent_history"])
    hist_recent["Date"] = pd.to_datetime(hist_recent["ds"]).dt.normalize()
    hist_recent["Vehicles Per Day"] = hist_recent["vehicles"].astype(float)
    hist_recent["Is Weekend"] = hist_recent["is_weekend"].astype(bool)

    traffic_full = load_daily_traffic()

    upcoming_dicts = [d for d in payload["forecast_days"] if not d.get("is_bridge_gap", False)]
    fc_py = pd.DataFrame(upcoming_dicts).rename(
        columns={"ensemble": "Ensemble", "dow": "Day of Week", "is_weekend": "Is Weekend"}
    )
    fc_py["ds"] = pd.to_datetime(fc_py["ds"]).dt.normalize()
    fc_py["Day of Week Num"] = fc_py["ds"].dt.day_of_week

    for df_opt in day_filters:
        for pr_opt in priorities:
            for td_opt in trip_days_list:
                key = f"{df_opt}|{pr_opt}|{td_opt}"
                js_rec = js_results[key]
                py_rec = compute_recommendations(
                    fc_py,
                    traffic_full,
                    day_filter=df_opt,
                    priority=pr_opt,
                    trip_days=td_opt,
                    cv_info=payload["cv_benchmark"],
                )
                assert js_rec is not None and py_rec is not None, f"Mismatch None on {key}"
                assert js_rec["bestDs"] == pd.Timestamp(py_rec.best_day["ds"]).strftime("%Y-%m-%d"), f"bestDs mismatch {key}"
                assert js_rec["worstDs"] == pd.Timestamp(py_rec.worst_day["ds"]).strftime("%Y-%m-%d"), f"worstDs mismatch {key}"
                assert js_rec["winStart"] == py_rec.best_window["start"], f"winStart mismatch {key}"
                assert js_rec["winEnd"] == py_rec.best_window["end"], f"winEnd mismatch {key}"
                assert js_rec["winDays"] == py_rec.best_window["actual_days"], f"winDays mismatch {key}"
                assert js_rec["trendLabel"] == py_rec.trend_label, f"trendLabel mismatch {key}"
                assert js_rec["busyCount"] == py_rec.busy_days_count, f"busyCount mismatch {key}"


def test_streamlit_app_lightweight_deps_and_smoke():
    """Verify traffic_app.py imports zero heavy ML packages and executes cleanly in AppTest."""
    app_path = PROJECT_ROOT / "traffic_app.py"
    tree = ast.parse(app_path.read_text(encoding="utf-8"))
    imported_roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".")[0])

    heavy_pkgs = {"torch", "neuralforecast", "lightgbm", "catboost", "statsforecast", "mlforecast", "pantab", "sklearn"}
    assert imported_roots.isdisjoint(heavy_pkgs), f"Heavy ML deps accidentally imported in traffic_app.py: {imported_roots & heavy_pkgs}"

    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(app_path), default_timeout=20).run()
    assert not at.exception
    assert not at.error
    assert len(at.tabs) == 4

    at.radio[0].set_value("Weekdays only").run()
    assert not at.exception and not at.error
    at.radio[1].set_value("Best Weather").run()
    assert not at.exception and not at.error
