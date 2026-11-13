"""Banff Visit Planner — Modernized Streamlit Companion Dashboard.

Lightweight runtime frontend synced with the daily forecasting pipeline
(`outputs/forecast_payload.json` + `predictions.csv` + `TW Traffic _data.csv`).
Strictly avoids heavy training/ML dependencies so Streamlit Community Cloud
runs cleanly off `requirements.txt`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from banff_planner.analytics import (  # noqa: E402
    ForecastSummary,
    compute_recommendations,
    enrich_forecast_analytics,
    filter_forecast_days,
)
from banff_planner.config import OUTPUTS_DIR, ROOT_PREDICTIONS_CSV, ROOT_TRAFFIC_TSV  # noqa: E402
from banff_planner.data_loader import load_daily_traffic  # noqa: E402

STATIC_WEB_URL = "https://v-datos.github.io/Banff_Visit_Planner_App/"

COLORS: Dict[str, str] = {
    "bg": "#0E1412",
    "surface": "#161E1B",
    "surface_alt": "#1E2824",
    "border": "#2A3732",
    "ink": "#F1F5F3",
    "muted": "#9BA8A2",
    "accent": "#34B395",
    "glacier": "#56B4E9",
    "history": "#94A3B8",
    "best": "#4ADE80",
    "good": "#A3E635",
    "warning": "#FBBF24",
    "bad": "#F87171",
}


def calgary_today() -> pd.Timestamp:
    """Current local calendar date in Banff, Alberta (Mountain Time)."""
    return pd.Timestamp.now(tz=ZoneInfo("America/Edmonton")).tz_localize(None).normalize()


def inject_styles() -> None:
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-color: {COLORS["bg"]};
            color: {COLORS["ink"]};
        }}
        .block-container {{
            max-width: 1220px;
            padding-top: 1.5rem;
            padding-bottom: 3rem;
        }}
        h1, h2, h3, h4, p, div, span, label {{
            color: {COLORS["ink"]};
        }}
        [data-testid="stSidebar"] {{
            background: {COLORS["surface"]};
            border-right: 1px solid {COLORS["border"]};
        }}
        .tabular {{
            font-variant-numeric: tabular-nums;
        }}
        .top-banner {{
            background: {COLORS["surface"]};
            border: 1px solid {COLORS["border"]};
            border-radius: 16px;
            padding: 1.15rem 1.35rem;
            margin-bottom: 1rem;
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
        }}
        .brand-h1 {{
            margin: 0;
            font-size: 2rem;
            font-weight: 800;
            letter-spacing: -0.025em;
            line-height: 1.1;
        }}
        .brand-sub {{
            color: {COLORS["muted"]};
            font-size: 0.92rem;
            margin: 0.35rem 0 0;
        }}
        .web-cta {{
            display: inline-block;
            background: rgba(52, 179, 149, 0.14);
            color: {COLORS["accent"]} !important;
            border: 1px solid rgba(52, 179, 149, 0.35);
            border-radius: 999px;
            padding: 0.45rem 0.95rem;
            font-size: 0.85rem;
            font-weight: 700;
            text-decoration: none;
            white-space: nowrap;
        }}
        .hero-grid {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 0.95rem;
            margin-bottom: 1rem;
        }}
        .panel-card {{
            background: {COLORS["surface"]};
            border: 1px solid {COLORS["border"]};
            border-radius: 16px;
            padding: 1.1rem 1.15rem;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }}
        .eyebrow {{
            font-size: 0.74rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: {COLORS["muted"]};
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.5rem;
        }}
        .card-date {{
            font-size: 1.55rem;
            font-weight: 800;
            letter-spacing: -0.02em;
            line-height: 1.15;
            margin-bottom: 0.4rem;
        }}
        .card-body {{
            color: {COLORS["muted"]};
            font-size: 0.88rem;
            line-height: 1.45;
            margin-top: 0.45rem;
        }}
        .signal-pill {{
            display: inline-block;
            border-radius: 999px;
            padding: 0.2rem 0.62rem;
            font-size: 0.76rem;
            font-weight: 700;
        }}
        .tone-best {{ background: rgba(74, 222, 128, 0.15); color: {COLORS["best"]}; }}
        .tone-good {{ background: rgba(163, 230, 53, 0.15); color: {COLORS["good"]}; }}
        .tone-warning {{ background: rgba(251, 191, 36, 0.15); color: {COLORS["warning"]}; }}
        .tone-bad {{ background: rgba(248, 113, 113, 0.15); color: {COLORS["bad"]}; }}
        .score-chip {{
            background: rgba(52, 179, 149, 0.15);
            color: {COLORS["accent"]};
            border-radius: 8px;
            padding: 0.18rem 0.5rem;
            font-size: 0.78rem;
            font-weight: 700;
        }}
        .mini-grid {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 0.85rem;
            margin-bottom: 1.1rem;
        }}
        .mini-stat {{
            background: {COLORS["surface"]};
            border: 1px solid {COLORS["border"]};
            border-radius: 12px;
            padding: 0.85rem 0.95rem;
        }}
        .mini-lbl {{
            font-size: 0.72rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: {COLORS["muted"]};
            margin-bottom: 0.2rem;
        }}
        .mini-val {{
            font-size: 1.02rem;
            font-weight: 700;
            margin-bottom: 0.15rem;
        }}
        .mini-sub {{
            font-size: 0.78rem;
            color: {COLORS["muted"]};
        }}
        .driver-pill {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: {COLORS["surface_alt"]};
            border: 1px solid {COLORS["border"]};
            border-radius: 8px;
            padding: 0.45rem 0.7rem;
            margin-bottom: 0.4rem;
            font-size: 0.84rem;
        }}
        @media (max-width: 960px) {{
            .hero-grid {{ grid-template-columns: 1fr; }}
            .mini-grid {{ grid-template-columns: 1fr 1fr; }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def load_payload_json(payload_path: Path = OUTPUTS_DIR / "forecast_payload.json") -> Optional[Dict[str, Any]]:
    if not payload_path.exists():
        return None
    try:
        return json.loads(payload_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def load_historical_data(csv_path: Path = ROOT_TRAFFIC_TSV) -> pd.DataFrame:
    """Load and validate historical daily Banff traffic counts."""
    return load_daily_traffic(csv_path=csv_path, reindex_daily=True)


def load_forecast_data(
    historical: pd.DataFrame,
    today: Optional[pd.Timestamp] = None,
    predictions_path: Path = ROOT_PREDICTIONS_CSV,
    full_csv_path: Path = OUTPUTS_DIR / "daily_forecast_full.csv",
    payload: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """Load upcoming daily forecast enriched with dual percentiles, weather, and conformal bounds."""
    today_ts = pd.Timestamp(today).normalize() if today is not None else calgary_today()

    if full_csv_path.exists():
        df = pd.read_csv(full_csv_path)
    elif predictions_path.exists():
        df = pd.read_csv(predictions_path)
    elif payload and payload.get("forecast_days"):
        df = pd.DataFrame(payload["forecast_days"]).rename(
            columns={"ensemble": "Ensemble", "dow": "Day of Week", "is_weekend": "Is Weekend"}
        )
    else:
        raise FileNotFoundError(f"Neither {full_csv_path} nor {predictions_path} exists.")

    df.columns = [str(c).strip() for c in df.columns]
    if "ds" not in df.columns or "Ensemble" not in df.columns:
        raise KeyError("Forecast file must include 'ds' and 'Ensemble' columns.")

    df["ds"] = pd.to_datetime(df["ds"], errors="coerce").dt.normalize()
    df["Ensemble"] = pd.to_numeric(df["Ensemble"], errors="coerce")
    df = df.dropna(subset=["ds", "Ensemble"]).sort_values("ds").reset_index(drop=True)

    # Filter strictly to upcoming calendar dates relative to Banff MT today
    upcoming = df.loc[df["ds"] >= today_ts].copy()
    if upcoming.empty:
        return upcoming

    # If reading minimal predictions.csv without full enrichment columns, compute them cleanly
    required_rich = {
        "overall_percentile",
        "daytype_percentile",
        "dow_seasonal_norm",
        "vs_dow_norm_pct",
        "weather_score",
        "visit_score_balanced",
    }
    if not required_rich.issubset(upcoming.columns):
        # Use stored conformal intervals if present, otherwise derive conservative spread around ensemble
        if "pi80_low" not in upcoming.columns and {"yhat_lower", "yhat_upper"}.issubset(upcoming.columns):
            upcoming["pi80_low"] = pd.to_numeric(upcoming["yhat_lower"], errors="coerce")
            upcoming["pi80_high"] = pd.to_numeric(upcoming["yhat_upper"], errors="coerce")
        upcoming = enrich_forecast_analytics(upcoming, historical, exog_panel=None)
    else:
        upcoming["Day of Week"] = upcoming["ds"].dt.day_name()
        upcoming["Day of Week Num"] = upcoming["ds"].dt.day_of_week
        upcoming["Is Weekend"] = upcoming["Day of Week Num"].isin([5, 6])

    # Ensure conformal bounds always exist for charting
    if "pi80_low" not in upcoming.columns:
        upcoming["pi80_low"] = upcoming["Ensemble"] * 0.90
        upcoming["pi80_high"] = upcoming["Ensemble"] * 1.10
    if "pi95_low" not in upcoming.columns:
        upcoming["pi95_low"] = upcoming["Ensemble"] * 0.84
        upcoming["pi95_high"] = upcoming["Ensemble"] * 1.16

    for num_col in (
        "pi80_low",
        "pi80_high",
        "pi95_low",
        "pi95_high",
        "temp_max_c",
        "temp_min_c",
        "precip_prob_pct",
        "precip_mm",
        "snowfall_cm",
        "weather_score",
        "west_entrance",
        "east_entrance",
        "mountain_ave",
    ):
        if num_col in upcoming.columns:
            upcoming[num_col] = pd.to_numeric(upcoming[num_col], errors="coerce")

    return upcoming.reset_index(drop=True)


def prepare_dashboard_state(
    historical: pd.DataFrame,
    forecast: pd.DataFrame,
    day_filter: str = "All upcoming days",
    priority: str = "balanced",
    trip_days: int = 3,
    cv_info: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, Optional[ForecastSummary]]:
    """Pure computation wrapper used by both Streamlit UI and automated unit tests."""
    filtered = filter_forecast_days(forecast, day_filter)
    summary = compute_recommendations(
        forecast,
        historical,
        day_filter=day_filter,
        priority=priority,
        trip_days=trip_days,
        cv_info=cv_info,
    )
    return filtered, summary


@st.cache_data(show_spinner=False, ttl=1800)
def _cached_load_all(today_str: str) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[Dict[str, Any]]]:
    payload = load_payload_json()
    historical = load_historical_data()
    forecast = load_forecast_data(historical, today=pd.Timestamp(today_str), payload=payload)
    return historical, forecast, payload


def _fmt_temp(v: Any) -> str:
    if v is None or pd.isna(v):
        return "—"
    return f"{round(float(v))}°C"


def _clean_str(v: Any) -> str:
    if v is None or pd.isna(v):
        return ""
    s = str(v).strip()
    return "" if s.lower() == "nan" else s


def _fmt_drivers_lookup(payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not payload or not payload.get("forecast_days"):
        return {}
    return {str(d["ds"]): d.get("driver_attributions", {}) for d in payload["forecast_days"]}


def _score_col_for(day_filter: str, priority: str) -> str:
    is_scoped = day_filter in ("Weekdays only", "Weekends only")
    if priority == "quiet":
        return "traffic_score_daytype" if is_scoped else "traffic_score_overall"
    if priority == "weather":
        return "visit_score_weather"
    return "visit_score_scoped_balanced" if is_scoped else "visit_score_balanced"


def render_hero_cards(
    summary: ForecastSummary,
    filtered: pd.DataFrame,
    day_filter: str,
    priority: str,
    payload: Optional[Dict[str, Any]],
) -> None:
    is_scoped = day_filter in ("Weekdays only", "Weekends only")
    score_col = _score_col_for(day_filter, priority)
    best = summary.best_day
    worst = summary.worst_day
    win = summary.best_window

    best_ds = pd.Timestamp(best["ds"])
    worst_ds = pd.Timestamp(worst["ds"])
    best_sig = best["signal_daytype"] if is_scoped else best["signal_overall"]
    best_tone = best["tone_daytype"] if is_scoped else best["tone_overall"]
    worst_sig = worst["signal_daytype"] if is_scoped else worst["signal_overall"]
    worst_tone = worst["tone_daytype"] if is_scoped else worst["tone_overall"]

    drivers_map = _fmt_drivers_lookup(payload)
    best_drivers = drivers_map.get(best_ds.strftime("%Y-%m-%d"), {}).get("drivers", [])
    best_driver_txt = best_drivers[0]["factor"] if best_drivers else "Calmer road demand relative to seasonal normal"

    best_wx = f"{_fmt_temp(best.get('temp_max_c'))} / {_fmt_temp(best.get('temp_min_c'))} · {best.get('weather_label', 'Seasonal')}"
    worst_wx = f"{_fmt_temp(worst.get('temp_max_c'))} / {_fmt_temp(worst.get('temp_min_c'))} · {worst.get('weather_label', 'Seasonal')}"
    best_ci = f"{int(round(best['pi80_low'])):,}–{int(round(best['pi80_high'])):,}"
    worst_event = _clean_str(worst.get("event_label"))

    bench = (payload or {}).get("cv_benchmark", {})
    cov80 = bench.get("empirical_cov80_pct", 80.0)
    mape = None
    for row in bench.get("summary_metrics", []):
        if row.get("model") == "Ensemble_Weighted":
            mape = row.get("MAPE")
            break

    win_avg = int(round(float(win["avg_vehicles"]))) if win.get("avg_vehicles") else 0

    st.markdown(
        f"""
        <div class="hero-grid">
          <div class="panel-card">
            <div>
              <div class="eyebrow"><span>Recommended Day</span><span class="score-chip tabular">Score {float(best[score_col]):.0f}/100</span></div>
              <div class="card-date">{best_ds.strftime("%a, %b %d")}</div>
              <span class="signal-pill tone-{best_tone}">{best_sig}</span>
              <div class="card-body">
                <strong>{_fmt_temp(best.get('temp_max_c'))}</strong> high · {best_wx}<br>
                <span class="tabular"><strong>{int(round(best['Ensemble'])):,} vehicles/day</strong> (80% CI: {best_ci})</span><br>
                Driver: {best_driver_txt}
              </div>
            </div>
            <div class="mini-sub tabular" style="margin-top:0.75rem;padding-top:0.6rem;border-top:1px solid {COLORS['border']};">
              {float(best['vs_dow_norm_pct']):+.1f}% vs typical {best_ds.strftime("%a")} · {float(best['daytype_percentile']):.0f}th day-type pct
            </div>
          </div>

          <div class="panel-card">
            <div>
              <div class="eyebrow"><span>Busiest Day to Avoid</span><span class="signal-pill tone-warning">{worst_event if worst_event else "Peak traffic"}</span></div>
              <div class="card-date">{worst_ds.strftime("%a, %b %d")}</div>
              <span class="signal-pill tone-{worst_tone}">{worst_sig}</span>
              <div class="card-body">
                {worst_wx}<br>
                <span class="tabular"><strong>{int(round(worst['Ensemble'])):,} vehicles/day</strong> ({float(worst['vs_dow_norm_pct']):+.1f}% vs typical {worst_ds.strftime("%a")})</span><br>
                Arrive before 08:45 AM or ride Roam Transit from Banff Train Station.
              </div>
            </div>
            <div class="mini-sub tabular" style="margin-top:0.75rem;padding-top:0.6rem;border-top:1px solid {COLORS['border']};">
              West {int(round(worst.get('west_entrance', 0))):,} · East {int(round(worst.get('east_entrance', 0))):,} vehicles
            </div>
          </div>

          <div class="panel-card">
            <div>
              <div class="eyebrow"><span>Best {win['actual_days']}-Day Contiguous Stay</span><span class="score-chip tabular">Avg {win_avg:,}/day</span></div>
              <div class="card-date" style="font-size:1.3rem;">{win['text']}</div>
              <div class="card-body">
                <strong>{summary.trend_label}:</strong> {summary.trend_detail}<br><br>
                {summary.confidence_detail}
              </div>
            </div>
            <div class="mini-sub tabular" style="margin-top:0.75rem;padding-top:0.6rem;border-top:1px solid {COLORS['border']};">
              {summary.busy_days_count} of {len(filtered)} matching days heavy · Scope: {summary.filtered_days_label}
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    mape_txt = f"{float(mape):.2f}% MAPE" if mape is not None else "Multi-model log ensemble"
    st.markdown(
        f"""
        <div class="mini-grid">
          <div class="mini-stat">
            <div class="mini-lbl">Contiguous Stay</div>
            <div class="mini-val tabular">{win['text']}</div>
            <div class="mini-sub">Strictly {win['actual_days']} consecutive calendar day(s)</div>
          </div>
          <div class="mini-stat">
            <div class="mini-lbl">Matched Trend</div>
            <div class="mini-val">{summary.trend_label}</div>
            <div class="mini-sub">Compared vs matching day-type &amp; seasonal norm</div>
          </div>
          <div class="mini-stat">
            <div class="mini-lbl">Conformal Precision</div>
            <div class="mini-val tabular">{summary.confidence_label}</div>
            <div class="mini-sub tabular">{mape_txt} · {cov80:.1f}% OOS 80% coverage</div>
          </div>
          <div class="mini-stat">
            <div class="mini-lbl">Planning Window</div>
            <div class="mini-val tabular">{len(filtered)} days</div>
            <div class="mini-sub tabular">{filtered['ds'].min().strftime('%b %d')} – {filtered['ds'].max().strftime('%b %d, %Y')}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_forecast_chart(
    historical: pd.DataFrame,
    filtered: pd.DataFrame,
    summary: ForecastSummary,
    show_history: bool,
    show_pi95: bool,
) -> None:
    fig = go.Figure()
    fc = filtered.sort_values("ds").copy()

    for _, row in fc.loc[fc["Is Weekend"]].iterrows():
        s0 = pd.Timestamp(row["ds"]).normalize() - pd.Timedelta(hours=12)
        s1 = pd.Timestamp(row["ds"]).normalize() + pd.Timedelta(hours=12)
        fig.add_vrect(x0=s0, x1=s1, fillcolor="rgba(251,191,36,0.06)", line_width=0, layer="below")

    if show_history:
        hist_tail = historical.tail(45)
        fig.add_trace(
            go.Scatter(
                x=hist_tail["Date"],
                y=hist_tail["Vehicles Per Day"],
                mode="lines",
                name="Observed (45d)",
                line=dict(color=COLORS["history"], width=2),
                opacity=0.55,
                hovertemplate="%{x|%a, %b %d}: %{y:,.0f} vehicles<extra></extra>",
            )
        )
        if not fc.empty and not hist_tail.empty:
            fig.add_trace(
                go.Scatter(
                    x=[hist_tail["Date"].iloc[-1], fc["ds"].iloc[0]],
                    y=[hist_tail["Vehicles Per Day"].iloc[-1], fc["Ensemble"].iloc[0]],
                    mode="lines",
                    line=dict(color=COLORS["glacier"], width=2, dash="dot"),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

    if show_pi95 and {"pi95_low", "pi95_high"}.issubset(fc.columns):
        fig.add_trace(
            go.Scatter(x=fc["ds"], y=fc["pi95_high"], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip")
        )
        fig.add_trace(
            go.Scatter(
                x=fc["ds"],
                y=fc["pi95_low"],
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor="rgba(86, 180, 233, 0.08)",
                name="95% Conformal PI",
                hoverinfo="skip",
            )
        )

    if {"pi80_low", "pi80_high"}.issubset(fc.columns):
        fig.add_trace(
            go.Scatter(x=fc["ds"], y=fc["pi80_high"], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip")
        )
        fig.add_trace(
            go.Scatter(
                x=fc["ds"],
                y=fc["pi80_low"],
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor="rgba(86, 180, 233, 0.18)",
                name="80% Conformal PI",
                hoverinfo="skip",
            )
        )

    fig.add_trace(
        go.Scatter(
            x=fc["ds"],
            y=fc["Ensemble"],
            mode="lines+markers",
            name="Ensemble Forecast",
            line=dict(color=COLORS["glacier"], width=3),
            marker=dict(size=8, color=COLORS["glacier"]),
            hovertemplate="%{x|%A, %b %d}<br>%{y:,.0f} vehicles/day<extra></extra>",
        )
    )

    for label, row, color, ay in [
        ("Best pick", summary.best_day, COLORS["best"], -44),
        ("Peak avoid", summary.worst_day, COLORS["bad"], -50),
    ]:
        fig.add_trace(
            go.Scatter(
                x=[row["ds"]],
                y=[row["Ensemble"]],
                mode="markers",
                showlegend=False,
                marker=dict(size=14, color=color, line=dict(color=COLORS["ink"], width=2)),
                hovertemplate="%{x|%A, %b %d}<br>%{y:,.0f} vehicles/day<extra></extra>",
            )
        )
        fig.add_annotation(
            x=row["ds"],
            y=row["Ensemble"],
            text=label,
            showarrow=True,
            arrowhead=2,
            ax=0,
            ay=ay,
            bgcolor=COLORS["surface"],
            bordercolor=color,
            font=dict(color=COLORS["ink"]),
        )

    fig.update_layout(
        height=430,
        margin=dict(l=10, r=10, t=24, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=COLORS["surface"],
        hovermode="x unified",
        legend=dict(orientation="h", y=1.05, x=0),
        xaxis_title="",
        yaxis_title="Vehicles per day (two-way entrances)",
        font=dict(color=COLORS["ink"]),
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor="rgba(155, 168, 162, 0.14)", zeroline=False)
    st.plotly_chart(fig, width="stretch")


def render_inspector_and_picks(
    filtered: pd.DataFrame,
    summary: ForecastSummary,
    day_filter: str,
    priority: str,
    payload: Optional[Dict[str, Any]],
) -> None:
    is_scoped = day_filter in ("Weekdays only", "Weekends only")
    score_col = _score_col_for(day_filter, priority)
    drivers_map = _fmt_drivers_lookup(payload)

    col_insp, col_picks = st.columns([1.25, 1.0])
    with col_insp:
        st.markdown("#### Selected Day Deep Dive")
        date_options = filtered["ds"].dt.strftime("%Y-%m-%d").tolist()
        best_ds_str = pd.Timestamp(summary.best_day["ds"]).strftime("%Y-%m-%d")
        default_idx = date_options.index(best_ds_str) if best_ds_str in date_options else 0
        sel_ds = st.selectbox(
            "Inspect day",
            options=date_options,
            index=default_idx,
            format_func=lambda s: pd.Timestamp(s).strftime("%a, %b %d (%Y)"),
            label_visibility="collapsed",
        )
        row = filtered.loc[filtered["ds"].dt.strftime("%Y-%m-%d") == sel_ds].iloc[0]

        m1, m2, m3 = st.columns(3)
        m1.metric(
            "Expected Traffic",
            f"{int(round(row['Ensemble'])):,}",
            f"{float(row['vs_dow_norm_pct']):+.1f}% vs {pd.Timestamp(row['ds']).strftime('%a')} norm",
        )
        m2.metric(
            "Weather & Comfort",
            f"{_fmt_temp(row.get('temp_max_c'))} / {_fmt_temp(row.get('temp_min_c'))}",
            f"{row.get('weather_label', 'Seasonal')} ({float(row['weather_score']):.0f}/100)",
        )
        m3.metric(
            "Visit Score",
            f"{float(row[score_col]):.0f} / 100",
            f"{float(row['daytype_percentile']):.0f}th day-type pct",
        )

        st.caption(
            f"**Dual Crowd Benchmarks:** {float(row['daytype_percentile']):.0f}th percentile vs same day-of-week "
            f"(3-yr norm: {int(round(row['dow_seasonal_norm'])):,}) · "
            f"{float(row['overall_percentile']):.0f}th percentile across all days in ±10d season. · "
            f"**80% Conformal CI:** {int(round(row['pi80_low'])):,}–{int(round(row['pi80_high'])):,} veh/day."
        )

        # Entrance corridor breakdown
        if pd.notna(row.get("west_entrance")) and pd.notna(row.get("east_entrance")):
            w_val = int(round(row["west_entrance"]))
            e_val = int(round(row["east_entrance"]))
            m_val = int(round(row["mountain_ave"])) if pd.notna(row.get("mountain_ave")) else 0
            st.caption(
                f"**Expected Corridor Split:** West (Norquay / Bow Valley) {w_val:,} · "
                f"East (Banff Ave / Hwy 1) {e_val:,} · Mountain Ave / Gondola ~{m_val:,}"
            )

        # Causal forecast drivers
        day_drivers = drivers_map.get(sel_ds, {}).get("drivers", [])
        if day_drivers:
            st.markdown("**Why this forecast?**")
            for dr in day_drivers:
                imp = float(dr.get("impact_pct", 0.0))
                tone = COLORS["warning"] if imp >= 0 else COLORS["best"]
                st.markdown(
                    f'<div class="driver-pill"><span>{dr.get("factor", "Seasonal rhythm")}</span>'
                    f'<strong class="tabular" style="color:{tone};">{imp:+.1f}%</strong></div>',
                    unsafe_allow_html=True,
                )

    with col_picks:
        st.markdown("#### Top Ranked Days to Visit")
        for rank, (_, r) in enumerate(summary.top_days.iterrows(), start=1):
            sig = r["signal_daytype"] if is_scoped else r["signal_overall"]
            tone = r["tone_daytype"] if is_scoped else r["tone_overall"]
            st.markdown(
                f'<div class="driver-pill">'
                f'<span><strong>#{rank} {pd.Timestamp(r["ds"]).strftime("%a, %b %d")}</strong> · '
                f'<span class="signal-pill tone-{tone}">{sig}</span><br>'
                f'<small style="color:{COLORS["muted"]};">{_fmt_temp(r.get("temp_max_c"))} {r.get("weather_label", "")}</small></span>'
                f'<strong class="tabular">{int(round(r["Ensemble"])):,} veh<br>'
                f'<small style="color:{COLORS["accent"]};">{float(r[score_col]):.0f}/100</small></strong>'
                f"</div>",
                unsafe_allow_html=True,
            )

        st.markdown("#### Days to Avoid")
        for rank, (_, r) in enumerate(summary.avoid_days.iterrows(), start=1):
            sig = r["signal_daytype"] if is_scoped else r["signal_overall"]
            tone = r["tone_daytype"] if is_scoped else r["tone_overall"]
            ev_txt = _clean_str(r.get("event_label"))
            ev = f" · {ev_txt}" if ev_txt else ""
            st.markdown(
                f'<div class="driver-pill">'
                f'<span><strong>#{rank} {pd.Timestamp(r["ds"]).strftime("%a, %b %d")}</strong> · '
                f'<span class="signal-pill tone-{tone}">{sig}</span><br>'
                f'<small style="color:{COLORS["muted"]};">{_fmt_temp(r.get("temp_max_c"))} {r.get("weather_label", "")}{ev}</small></span>'
                f'<strong class="tabular">{int(round(r["Ensemble"])):,} veh</strong>'
                f"</div>",
                unsafe_allow_html=True,
            )


def render_deep_dive_tabs(
    historical: pd.DataFrame,
    filtered: pd.DataFrame,
    day_filter: str,
    priority: str,
    payload: Optional[Dict[str, Any]],
) -> None:
    is_scoped = day_filter in ("Weekdays only", "Weekends only")
    score_col = _score_col_for(day_filter, priority)

    tab_tbl, tab_wk, tab_yr, tab_cv = st.tabs(
        ["Full 15-Day Table", "Weekly Rhythm", "3-Year Seasonality", "Model Accuracy & Sources"]
    )

    with tab_tbl:
        disp = pd.DataFrame(
            {
                "Date": filtered["ds"].dt.strftime("%Y-%m-%d"),
                "Day": filtered["Day of Week"].str[:3],
                "Conditions": filtered.get("weather_label", pd.Series("Seasonal", index=filtered.index)),
                "High / Low (°C)": [
                    f"{_fmt_temp(mx)} / {_fmt_temp(mn)}"
                    for mx, mn in zip(filtered.get("temp_max_c", [np.nan] * len(filtered)), filtered.get("temp_min_c", [np.nan] * len(filtered)))
                ],
                "Vehicles/Day": filtered["Ensemble"].round().astype(int),
                "80% Conformal CI": [
                    f"{int(round(lo)):,}–{int(round(hi)):,}"
                    for lo, hi in zip(filtered["pi80_low"], filtered["pi80_high"])
                ],
                "Visit Score": filtered[score_col].round(0).astype(int),
                "Crowd Signal": filtered["signal_daytype"] if is_scoped else filtered["signal_overall"],
            }
        )
        if "event_label" in filtered.columns:
            disp["Event"] = filtered["event_label"].astype("string").fillna("")
        st.dataframe(disp, hide_index=True, width="stretch")

    with tab_wk:
        order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        recent = historical.tail(min(1095, len(historical)))
        stats = recent.groupby("Day of Week")["Vehicles Per Day"].mean().reindex(order).reset_index()
        colors = [COLORS["accent"] if d in order[:5] else COLORS["warning"] for d in stats["Day of Week"]]
        fig_dow = go.Figure(
            go.Bar(
                x=stats["Day of Week"],
                y=stats["Vehicles Per Day"],
                marker_color=colors,
                text=[f"{int(round(v)):,}" for v in stats["Vehicles Per Day"]],
                textposition="outside",
            )
        )
        fig_dow.update_layout(
            height=340,
            margin=dict(l=10, r=10, t=20, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor=COLORS["surface"],
            yaxis_title="3-Yr Avg Vehicles/Day",
            font=dict(color=COLORS["ink"]),
        )
        st.plotly_chart(fig_dow, width="stretch")

    with tab_yr:
        recent_years = sorted(historical["Year"].unique())[-3:]
        monthly = (
            historical.loc[historical["Year"].isin(recent_years)]
            .groupby(["Year", "Month"])["Vehicles Per Day"]
            .mean()
            .reset_index()
        )
        fig_mo = go.Figure()
        for yr in recent_years:
            y_df = monthly.loc[monthly["Year"] == yr]
            fig_mo.add_trace(go.Scatter(x=y_df["Month"], y=y_df["Vehicles Per Day"], mode="lines+markers", name=str(yr)))
        fig_mo.update_layout(
            height=340,
            margin=dict(l=10, r=10, t=20, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor=COLORS["surface"],
            yaxis_title="Average Vehicles/Day",
            xaxis=dict(
                tickmode="array",
                tickvals=list(range(1, 13)),
                ticktext=["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
            ),
            font=dict(color=COLORS["ink"]),
        )
        st.plotly_chart(fig_mo, width="stretch")

    with tab_cv:
        cv = (payload or {}).get("cv_benchmark", {})
        fresh = (payload or {}).get("data_freshness", {})
        summary_rows = cv.get("summary_metrics")
        if summary_rows:
            st.markdown(
                f"**{cv.get('n_folds', 12)}-Fold Expanding Out-of-Sample Benchmark "
                f"({cv.get('n_oof_days', 192)} days across all 4 Banff seasons)** — "
                f"Conformal coverage: 80% -> `{cv.get('empirical_cov80_pct', 80.0)}%`, "
                f"95% -> `{cv.get('empirical_cov95_pct', 95.0)}%`"
            )
            cv_df = pd.DataFrame(summary_rows)
            for col in ("MAE", "RMSE"):
                if col in cv_df.columns:
                    cv_df[col] = cv_df[col].round(0).astype(int)
            for col in ("MAPE", "sMAPE", "Bias_pct"):
                if col in cv_df.columns:
                    cv_df[col] = cv_df[col].round(2)
            if "MASE" in cv_df.columns:
                cv_df["MASE"] = cv_df["MASE"].round(3)
            st.dataframe(cv_df, hide_index=True, width="stretch")

        if fresh:
            st.markdown(
                f"- **Traffic Counters:** `{fresh.get('traffic_start_date')}` to `{fresh.get('traffic_last_observed')}` ({fresh.get('traffic_days', '—')} days)\n"
                f"- **Banff Weather & 16-Day NWP:** through `{fresh.get('weather_forecast_through')}` (ECCC BANFF CS + Open-Meteo)\n"
                f"- **Bank of Canada USD/CAD FX:** observed through `{fresh.get('fx_last_observed')}`\n"
                f"- **StatCan & Alberta Economic Dashboard:** through `{fresh.get('statcan_macro_through')}` (Hwy 1 & Park Attendance integrated; Zero Waste data strictly excluded)"
            )
        else:
            st.info(
                "Run `banff-planner` to populate `outputs/forecast_payload.json` with 12-fold CV benchmarks and live API freshness metadata."
            )


def main() -> None:
    st.set_page_config(page_title="Banff Visit Planner", page_icon="⛰️", layout="wide")
    inject_styles()

    today_mt = calgary_today()
    try:
        historical, forecast, payload = _cached_load_all(today_mt.strftime("%Y-%m-%d"))
    except Exception as exc:
        st.error(f"Unable to load Banff planner data: {exc}")
        st.stop()

    if forecast.empty:
        st.warning(
            "Published predictions have expired relative to today's Banff date. Run `banff-planner --refresh-web` or trigger the daily GitHub Actions update."
        )
        st.stop()

    with st.sidebar:
        st.markdown("### Trip Filters")
        day_filter = st.radio("Schedule", ["All upcoming days", "Weekdays only", "Weekends only"], index=0)
        priority_label = st.radio("Optimize for", ["Balanced", "Least Traffic", "Best Weather"], index=0)
        priority_map = {"Balanced": "balanced", "Least Traffic": "quiet", "Best Weather": "weather"}
        priority = priority_map[priority_label]
        trip_days = st.slider("Trip duration (days)", min_value=1, max_value=4, value=3)
        st.markdown("---")
        show_history = st.toggle("Show 45-day history", value=True)
        show_pi95 = st.toggle("Show 95% conformal band", value=False)
        st.markdown("---")
        st.caption(
            f"Forecast window: {forecast['ds'].min().strftime('%b %d')} – {forecast['ds'].max().strftime('%b %d, %Y')} ({len(forecast)} days)"
        )
        st.markdown(f"[Open fast GitHub Pages app]({STATIC_WEB_URL})")

    cv_info = (payload or {}).get("cv_benchmark")
    filtered, summary = prepare_dashboard_state(
        historical,
        forecast,
        day_filter=day_filter,
        priority=priority,
        trip_days=trip_days,
        cv_info=cv_info,
    )
    if summary is None or filtered.empty:
        st.warning("No upcoming forecast days match the selected filter.")
        st.stop()

    fresh = (payload or {}).get("data_freshness", {})
    last_obs_str = fresh.get("traffic_last_observed") or historical["Date"].max().strftime("%Y-%m-%d")
    st.markdown(
        f"""
        <div class="top-banner">
          <div>
            <h1 class="brand-h1">Banff Visit Planner</h1>
            <p class="brand-sub tabular">Log-scale multi-model ensemble + 16-day mountain weather + calibrated conformal bounds · Observed traffic through {last_obs_str}</p>
          </div>
          <a class="web-cta" href="{STATIC_WEB_URL}" target="_blank" rel="noopener">Open Instant Static Web App ↗</a>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_hero_cards(summary, filtered, day_filter, priority, payload)

    st.markdown("#### Daily Traffic Trajectory & Conformal Uncertainty")
    st.caption("Weekends are subtly shaded. Shaded bands depict calibrated 80% / 95% out-of-sample conformal prediction intervals.")
    render_forecast_chart(historical, filtered, summary, show_history=show_history, show_pi95=show_pi95)

    render_inspector_and_picks(filtered, summary, day_filter, priority, payload)

    st.markdown("---")
    render_deep_dive_tabs(historical, filtered, day_filter, priority, payload)


if __name__ == "__main__":
    main()
