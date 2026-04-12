from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data_utils import load_csv, normalize_historical_traffic


st.set_page_config(page_title="Banff Traffic Planner", page_icon="🏔️", layout="wide")

DATA_DIR = Path(__file__).resolve().parent
HISTORICAL_PATH = DATA_DIR / "TW Traffic _data.csv"
FORECAST_PATH = DATA_DIR / "predictions.csv"

COLORS = {
    "bg": "#0F172A",
    "card": "#111827",
    "card_alt": "#1F2937",
    "border": "#374151",
    "text": "#F9FAFB",
    "secondary": "#9CA3AF",
    "muted": "#6B7280",
    "best": "#22C55E",
    "good": "#84CC16",
    "warning": "#F59E0B",
    "bad": "#EF4444",
    "blue": "#38BDF8",
    "history": "#94A3B8",
}


@dataclass
class ForecastSummary:
    best_day: pd.Series
    worst_day: pd.Series
    top_days: pd.DataFrame
    avoid_days: pd.DataFrame
    recommendation: str
    best_window_text: str
    confidence_label: str
    confidence_detail: str
    trend_label: str
    trend_detail: str
    busy_days_count: int
    filtered_days_label: str
    weekly_insight: str
    seasonal_insight: str


def inject_styles() -> None:
    st.markdown(
        f"""
        <style>
        .stApp {{
            background:
                radial-gradient(circle at top right, rgba(56,189,248,0.10), transparent 22%),
                radial-gradient(circle at top left, rgba(34,197,94,0.08), transparent 18%),
                {COLORS["bg"]};
            color: {COLORS["text"]};
        }}
        .block-container {{
            max-width: 1180px;
            padding-top: 1.8rem;
            padding-bottom: 3rem;
        }}
        h1, h2, h3, h4, p, div, span, label {{
            color: {COLORS["text"]};
        }}
        [data-testid="stSidebar"] {{
            background: {COLORS["card"]};
            border-left: 1px solid {COLORS["border"]};
        }}
        .hero-shell {{
            background: linear-gradient(145deg, rgba(17,24,39,0.96), rgba(31,41,55,0.98));
            border: 1px solid {COLORS["border"]};
            border-radius: 26px;
            padding: 1.4rem 1.5rem;
            margin-bottom: 1rem;
            box-shadow: 0 18px 48px rgba(0,0,0,0.35);
        }}
        .eyebrow {{
            font-size: 0.76rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            color: {COLORS["secondary"]};
            margin-bottom: 0.4rem;
        }}
        .hero-title {{
            margin: 0;
            font-size: 2.8rem;
            line-height: 0.98;
            font-weight: 800;
        }}
        .hero-sub {{
            color: {COLORS["secondary"]};
            font-size: 1.02rem;
            margin-top: 0.65rem;
            margin-bottom: 0;
            max-width: 760px;
        }}
        .hero-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr 0.9fr;
            gap: 1rem;
            margin-top: 1rem;
            margin-bottom: 1rem;
        }}
        .decision-card {{
            border-radius: 24px;
            padding: 1.3rem 1.2rem;
            min-height: 220px;
            border: 1px solid {COLORS["border"]};
            background: linear-gradient(180deg, rgba(17,24,39,0.92), rgba(31,41,55,0.98));
            box-shadow: 0 18px 36px rgba(0,0,0,0.28);
        }}
        .decision-best {{
            border-color: rgba(34,197,94,0.45);
            box-shadow: 0 0 0 1px rgba(34,197,94,0.18), 0 0 28px rgba(34,197,94,0.18);
            background: linear-gradient(180deg, rgba(16,35,28,0.96), rgba(17,24,39,0.98));
        }}
        .decision-avoid {{
            border-color: rgba(239,68,68,0.45);
            box-shadow: 0 0 0 1px rgba(239,68,68,0.18), 0 0 28px rgba(239,68,68,0.16);
            background: linear-gradient(180deg, rgba(48,23,25,0.96), rgba(17,24,39,0.98));
        }}
        .decision-kpi {{
            min-height: 220px;
        }}
        .card-label {{
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.11em;
            color: {COLORS["secondary"]};
            margin-bottom: 0.8rem;
        }}
        .card-date {{
            font-size: 2rem;
            line-height: 1.0;
            font-weight: 800;
            margin-bottom: 0.45rem;
        }}
        .card-status {{
            font-size: 1rem;
            font-weight: 700;
            margin-bottom: 0.35rem;
        }}
        .card-metric {{
            font-size: 1rem;
            color: {COLORS["text"]};
            margin-bottom: 0.45rem;
        }}
        .card-detail {{
            color: {COLORS["secondary"]};
            font-size: 0.95rem;
            line-height: 1.4;
        }}
        .banner {{
            border-radius: 18px;
            padding: 1rem 1.05rem;
            background: linear-gradient(90deg, rgba(34,197,94,0.20), rgba(56,189,248,0.14));
            border: 1px solid rgba(56,189,248,0.22);
            margin-bottom: 1.1rem;
        }}
        .banner strong {{
            color: {COLORS["text"]};
        }}
        .small-card-grid {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 0.9rem;
            margin-bottom: 1.4rem;
        }}
        .mini-card {{
            background: rgba(17,24,39,0.9);
            border: 1px solid {COLORS["border"]};
            border-radius: 18px;
            padding: 0.95rem 1rem;
        }}
        .mini-label {{
            color: {COLORS["secondary"]};
            font-size: 0.76rem;
            text-transform: uppercase;
            letter-spacing: 0.11em;
            margin-bottom: 0.35rem;
        }}
        .mini-value {{
            font-size: 1.08rem;
            font-weight: 700;
            margin-bottom: 0.18rem;
        }}
        .mini-detail {{
            color: {COLORS["secondary"]};
            font-size: 0.9rem;
        }}
        .section-shell {{
            background: rgba(17,24,39,0.88);
            border: 1px solid {COLORS["border"]};
            border-radius: 22px;
            padding: 1rem 1rem 0.8rem 1rem;
            box-shadow: 0 14px 32px rgba(0,0,0,0.24);
        }}
        .pill-row {{
            display: flex;
            gap: 0.85rem;
            overflow-x: auto;
            padding-bottom: 0.3rem;
            margin-top: 0.5rem;
        }}
        .pick-pill {{
            min-width: 100%;
            border-radius: 18px;
            padding: 0.95rem 1rem;
            border: 1px solid {COLORS["border"]};
            background: rgba(31,41,55,0.96);
            box-shadow: 0 8px 20px rgba(0,0,0,0.24);
        }}
        .pick-good {{
            box-shadow: 0 0 0 1px rgba(34,197,94,0.16), 0 0 20px rgba(34,197,94,0.10);
        }}
        .pick-bad {{
            box-shadow: 0 0 0 1px rgba(239,68,68,0.16), 0 0 20px rgba(239,68,68,0.10);
        }}
        .pick-rank {{
            color: {COLORS["secondary"]};
            font-size: 0.74rem;
            text-transform: uppercase;
            letter-spacing: 0.10em;
            margin-bottom: 0.35rem;
        }}
        .pick-date {{
            font-size: 1.08rem;
            font-weight: 800;
            margin-bottom: 0.3rem;
        }}
        .pill-badge {{
            display: inline-block;
            border-radius: 999px;
            padding: 0.18rem 0.55rem;
            font-size: 0.76rem;
            font-weight: 800;
            margin-bottom: 0.45rem;
        }}
        .badge-best {{
            background: rgba(34,197,94,0.18);
            color: {COLORS["best"]};
        }}
        .badge-good {{
            background: rgba(132,204,22,0.18);
            color: {COLORS["good"]};
        }}
        .badge-warning {{
            background: rgba(245,158,11,0.18);
            color: {COLORS["warning"]};
        }}
        .badge-bad {{
            background: rgba(239,68,68,0.18);
            color: {COLORS["bad"]};
        }}
        .insight-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1rem;
            margin-bottom: 1rem;
        }}
        .insight-card {{
            background: rgba(17,24,39,0.92);
            border: 1px solid {COLORS["border"]};
            border-radius: 18px;
            padding: 1rem;
        }}
        .insight-title {{
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.11em;
            color: {COLORS["secondary"]};
            margin-bottom: 0.45rem;
        }}
        .insight-body {{
            font-size: 1.03rem;
            font-weight: 700;
            line-height: 1.35;
        }}
        .muted {{
            color: {COLORS["secondary"]};
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def prepare_historical() -> pd.DataFrame:
    df = load_csv(HISTORICAL_PATH, encoding="utf-16", sep="\t")
    df = normalize_historical_traffic(df)
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Month Name"] = df["Date"].dt.strftime("%b")
    df["Day of Week"] = df["Date"].dt.day_name()
    df["Day of Week Num"] = df["Date"].dt.dayofweek
    df["Is Weekend"] = df["Day of Week Num"].isin([5, 6])
    return df


@st.cache_data(show_spinner=False)
def prepare_forecast() -> pd.DataFrame:
    df = load_csv(FORECAST_PATH)
    df.columns = [col.strip() for col in df.columns]
    required_cols = {"ds", "Ensemble"}
    if not required_cols.issubset(df.columns):
        missing = sorted(required_cols.difference(df.columns))
        raise KeyError(f"Missing columns in forecast data: {', '.join(missing)}")
    df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
    df["Ensemble"] = pd.to_numeric(df["Ensemble"], errors="coerce")
    df = df.dropna(subset=["ds", "Ensemble"]).sort_values("ds")
    df["Day of Week"] = df["ds"].dt.day_name()
    df["Day of Week Num"] = df["ds"].dt.dayofweek
    df["Is Weekend"] = df["Day of Week Num"].isin([5, 6])
    for col in ["yhat_lower", "yhat_upper"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def label_for_volume(volume: float, historical: pd.DataFrame) -> tuple[str, str]:
    q30 = historical["Vehicles Per Day"].quantile(0.30)
    q55 = historical["Vehicles Per Day"].quantile(0.55)
    q80 = historical["Vehicles Per Day"].quantile(0.80)
    if volume <= q30:
        return "Best day to visit", "best"
    if volume <= q55:
        return "Good day to visit", "good"
    if volume <= q80:
        return "Busy day", "warning"
    return "Avoid if possible", "bad"


def compute_traffic_score(forecast: pd.DataFrame, historical: pd.DataFrame) -> pd.Series:
    avg = historical["Vehicles Per Day"].mean()
    std = historical["Vehicles Per Day"].std()
    if pd.isna(std) or std == 0:
        return pd.Series(50.0, index=forecast.index)
    score = 50 - (((forecast["Ensemble"] - avg) / std) * 25)
    return score.clip(lower=0, upper=100).round(1)


def filter_forecast_days(forecast: pd.DataFrame, day_filter: str) -> pd.DataFrame:
    if day_filter == "Weekdays only":
        return forecast.loc[~forecast["Is Weekend"]].copy()
    if day_filter == "Weekends only":
        return forecast.loc[forecast["Is Weekend"]].copy()
    return forecast.copy()


def get_confidence_message(forecast: pd.DataFrame) -> tuple[str, str]:
    if {"yhat_lower", "yhat_upper"}.issubset(forecast.columns):
        width = (forecast["yhat_upper"] - forecast["yhat_lower"]).median()
        center = forecast["Ensemble"].median()
        relative = width / center if center else 0
        if relative < 0.15:
            return "High", "Forecast spread is tight across the next two weeks."
        if relative < 0.30:
            return "Moderate", "The outlook is stable, with normal day-to-day uncertainty."
        return "Lower", "Expect more variation than usual, especially on busy days."
    return "Moderate", "Based on model output without published interval bands."


def get_trend_message(historical: pd.DataFrame, forecast: pd.DataFrame) -> tuple[str, str]:
    recent = historical.tail(min(30, len(historical)))["Vehicles Per Day"].mean()
    next_week = forecast.head(min(7, len(forecast)))["Ensemble"].mean()
    if pd.isna(recent) or recent == 0:
        return "Steady", "Not enough recent history to compare."
    delta = ((next_week - recent) / recent) * 100
    if delta <= -8:
        return "Quieter trend", f"About {abs(delta):.0f}% below the recent average."
    if delta >= 8:
        return "Busier trend", f"About {delta:.0f}% above the recent average."
    return "Stable trend", "The next week looks close to recent traffic levels."


def compute_best_window(forecast: pd.DataFrame) -> str:
    if len(forecast) < 3:
        return forecast.iloc[0]["ds"].strftime("%a, %b %d")
    window = forecast[["ds", "Ensemble"]].copy()
    window["rolling_mean"] = window["Ensemble"].rolling(3).mean()
    best = window.dropna().sort_values("rolling_mean").iloc[0]
    idx = window.index.get_loc(best.name)
    start = window.iloc[idx - 2]["ds"]
    end = window.iloc[idx]["ds"]
    return f"{start.strftime('%a, %b %d')} to {end.strftime('%a, %b %d')}"


def build_insights(historical: pd.DataFrame) -> tuple[str, str]:
    weekday_avg = historical.groupby("Day of Week")["Vehicles Per Day"].mean().reindex(
        ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    )
    best_day = weekday_avg.idxmin()
    worst_day = weekday_avg.idxmax()
    diff_pct = ((weekday_avg[worst_day] - weekday_avg[best_day]) / weekday_avg[worst_day]) * 100

    month_avg = historical.groupby("Month")["Vehicles Per Day"].mean()
    peak_month = int(month_avg.idxmax())
    quiet_month = int(month_avg.idxmin())
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    weekly = f"{best_day}s are typically the quietest, with about {diff_pct:.0f}% less traffic than {worst_day}s."
    seasonal = f"{month_names[peak_month - 1]} is usually the busiest month, while {month_names[quiet_month - 1]} is the calmest."
    return weekly, seasonal


def compute_recommendations(forecast: pd.DataFrame, historical: pd.DataFrame, day_filter: str) -> ForecastSummary | None:
    filtered = filter_forecast_days(forecast, day_filter)
    if filtered.empty:
        return None

    scored = filtered.copy()
    scored["Traffic Score"] = compute_traffic_score(scored, historical)
    scored = scored.sort_values(["Traffic Score", "Ensemble"], ascending=[False, True])
    best_day = scored.iloc[0]
    worst_day = scored.sort_values(["Traffic Score", "Ensemble"], ascending=[True, False]).iloc[0]
    top_days = scored.head(min(3, len(scored)))
    avoid_days = scored.sort_values(["Traffic Score", "Ensemble"], ascending=[True, False]).head(min(3, len(scored)))
    confidence_label, confidence_detail = get_confidence_message(filtered)
    trend_label, trend_detail = get_trend_message(historical, filtered.sort_values("ds"))
    weekly_insight, seasonal_insight = build_insights(historical)
    busy_threshold = historical["Vehicles Per Day"].quantile(0.80)
    busy_days_count = int((filtered["Ensemble"] >= busy_threshold).sum())
    scope = "weekdays" if day_filter == "Weekdays only" else "weekends" if day_filter == "Weekends only" else "days"
    recommendation = (
        f"Go on {best_day['ds'].strftime('%A, %b %d')}. "
        f"Avoid {worst_day['ds'].strftime('%A, %b %d')} if your dates are flexible."
    )

    return ForecastSummary(
        best_day=best_day,
        worst_day=worst_day,
        top_days=top_days,
        avoid_days=avoid_days,
        recommendation=recommendation,
        best_window_text=compute_best_window(filtered.sort_values("ds")),
        confidence_label=confidence_label,
        confidence_detail=confidence_detail,
        trend_label=trend_label,
        trend_detail=trend_detail,
        busy_days_count=busy_days_count,
        filtered_days_label=scope,
        weekly_insight=weekly_insight,
        seasonal_insight=seasonal_insight,
    )


def render_decision_card(title: str, emoji: str, day_text: str, status: str, vehicles: int, detail: str, kind: str) -> str:
    klass = "decision-card decision-best" if kind == "best" else "decision-card decision-avoid"
    return (
        f'<div class="{klass}">'
        f'<div class="card-label">{emoji} {title}</div>'
        f'<div class="card-date">{day_text}</div>'
        f'<div class="card-status">{status}</div>'
        f'<div class="card-metric">🚗 {vehicles:,} vehicles</div>'
        f'<div class="card-detail">{detail}</div>'
        "</div>"
    )


def render_hero(summary: ForecastSummary, historical: pd.DataFrame) -> None:
    best_label, _ = label_for_volume(summary.best_day["Ensemble"], historical)
    worst_label, _ = label_for_volume(summary.worst_day["Ensemble"], historical)
    hero_html = (
        '<div class="hero-shell">'
        '<h1 class="hero-title">Banff Visit Planner</h1>'
        '<p class="hero-sub">This forecast is designed for one decision: pick the quietest upcoming day to visit Banff.</p>'
        '<div class="hero-grid">'
        + render_decision_card(
            "Best Day to Visit",
            "🌟",
            summary.best_day["ds"].strftime("%a, %b %d"),
            best_label,
            int(summary.best_day["Ensemble"]),
            "Lowest predicted traffic in the current planning window.",
            "best",
        )
        + render_decision_card(
            "Day to Avoid",
            "⚠️",
            summary.worst_day["ds"].strftime("%a, %b %d"),
            worst_label,
            int(summary.worst_day["Ensemble"]),
            "Highest predicted traffic in the current planning window.",
            "avoid",
        )
        + (
            '<div class="decision-card decision-kpi">'
            '<div class="card-label">📌 Decision Snapshot</div>'
            f'<div class="card-date">{summary.best_window_text}</div>'
            f'<div class="card-status">{summary.trend_label}</div>'
            f'<div class="card-metric">{summary.busy_days_count} busy {summary.filtered_days_label}</div>'
            f'<div class="card-detail">{summary.confidence_label} confidence. {summary.trend_detail}</div>'
            "</div>"
        )
        + "</div>"
        f'<div class="banner"><strong>Recommendation:</strong> {summary.recommendation}<br><span class="muted">Best low-traffic window: {summary.best_window_text}. {summary.confidence_detail}</span></div>'
        "</div>"
    )
    st.markdown(hero_html, unsafe_allow_html=True)


def render_mini_metrics(summary: ForecastSummary, forecast: pd.DataFrame) -> None:
    html = (
        '<div class="small-card-grid">'
        '<div class="mini-card">'
        '<div class="mini-label">Recent Trend (30 days)</div>'
        f'<div class="mini-value">{summary.trend_label}</div>'
        f'<div class="mini-detail">{summary.trend_detail}</div>'
        "</div>"
        '<div class="mini-card">'
        '<div class="mini-label">Forecast Window</div>'
        f'<div class="mini-value">{len(forecast)} days</div>'
        f'<div class="mini-detail">{forecast["ds"].min().strftime("%b %d")} to {forecast["ds"].max().strftime("%b %d, %Y")}</div>'
        "</div>"
        '<div class="mini-card">'
        '<div class="mini-label">Low-Traffic Period</div>'
        f'<div class="mini-value">{summary.best_window_text}</div>'
        '<div class="mini-detail">Strongest short window for a quieter visit.</div>'
        "</div>"
        "</div>"
    )
    st.markdown(html, unsafe_allow_html=True)


def add_weekend_shading(fig: go.Figure, forecast: pd.DataFrame) -> None:
    for _, row in forecast.loc[forecast["Is Weekend"]].iterrows():
        start = row["ds"].normalize() - pd.Timedelta(hours=12)
        end = row["ds"].normalize() + pd.Timedelta(hours=12)
        fig.add_vrect(x0=start, x1=end, fillcolor="rgba(245,158,11,0.08)", line_width=0, layer="below")


def render_forecast_chart(historical: pd.DataFrame, forecast: pd.DataFrame, summary: ForecastSummary, show_history: bool) -> None:
    fig = go.Figure()
    forecast = forecast.sort_values("ds")
    add_weekend_shading(fig, forecast)

    if show_history:
        history_tail = historical.tail(45)
        fig.add_trace(
            go.Scatter(
                x=history_tail["Date"],
                y=history_tail["Vehicles Per Day"],
                mode="lines",
                name="Recent history",
                line=dict(color=COLORS["history"], width=2),
                opacity=0.45,
                hovertemplate="%{x|%b %d}: %{y:,.0f} vehicles<extra></extra>",
            )
        )

    if {"yhat_lower", "yhat_upper"}.issubset(forecast.columns):
        fig.add_trace(
            go.Scatter(
                x=forecast["ds"],
                y=forecast["yhat_upper"],
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=forecast["ds"],
                y=forecast["yhat_lower"],
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor="rgba(56,189,248,0.12)",
                name="Forecast range",
                hoverinfo="skip",
            )
        )

    fig.add_trace(
        go.Scatter(
            x=forecast["ds"],
            y=forecast["Ensemble"],
            mode="lines+markers",
            name="Forecast",
            line=dict(color=COLORS["blue"], width=3),
            marker=dict(size=8, color=COLORS["blue"]),
            hovertemplate="%{x|%A, %b %d}<br>%{y:,.0f} vehicles<extra></extra>",
        )
    )

    for label, row, color, ay in [
        ("Best day", summary.best_day, COLORS["best"], -46),
        ("Avoid", summary.worst_day, COLORS["bad"], -56),
    ]:
        fig.add_trace(
            go.Scatter(
                x=[row["ds"]],
                y=[row["Ensemble"]],
                mode="markers",
                showlegend=False,
                marker=dict(size=15, color=color, line=dict(color="#F9FAFB", width=2)),
                hovertemplate="%{x|%A, %b %d}<br>%{y:,.0f} vehicles<extra></extra>",
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
            bgcolor="#111827",
            bordercolor=color,
            font=dict(color=COLORS["text"]),
        )

    fig.update_layout(
        height=470,
        margin=dict(l=10, r=10, t=20, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(17,24,39,0.74)",
        hovermode="x unified",
        legend=dict(orientation="h", y=1.04, x=0),
        xaxis_title="",
        yaxis_title="Vehicles per day",
        font=dict(color=COLORS["text"]),
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor="rgba(148,163,184,0.12)", zeroline=False)
    st.plotly_chart(fig, width="stretch")


def render_pick_pills(days: pd.DataFrame, historical: pd.DataFrame, title: str, mode: str) -> None:
    st.markdown(f"#### {title}")
    columns = st.columns(len(days)) if len(days) > 0 else []
    for rank, ((_, row), column) in enumerate(zip(days.iterrows(), columns), start=1):
        label, tone = label_for_volume(row["Ensemble"], historical)
        badge_class = {
            "best": "badge-best",
            "good": "badge-good",
            "warning": "badge-warning",
            "bad": "badge-bad",
        }[tone]
        pill_class = "pick-pill pick-good" if mode == "good" else "pick-pill pick-bad"
        with column:
            st.markdown(
                f"""
                <div class="{pill_class}">
                    <div class="pick-rank">#{rank}</div>
                    <div class="pick-date">{row["ds"].strftime("%a, %b %d")}</div>
                    <div class="pill-badge {badge_class}">{label}</div>
                    <div>🚗 {int(row["Ensemble"]):,} vehicles</div>
                    <div class="muted">{row["Day of Week"]}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_insight_cards(summary: ForecastSummary) -> None:
    st.markdown(
        f"""
        <div class="insight-grid">
            <div class="insight-card">
                <div class="insight-title">Weekly Insight</div>
                <div class="insight-body">💡 {summary.weekly_insight}</div>
            </div>
            <div class="insight-card">
                <div class="insight-title">Seasonal Tip</div>
                <div class="insight-body">📅 {summary.seasonal_insight}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_day_of_week_patterns(historical: pd.DataFrame) -> None:
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    stats = historical.groupby("Day of Week")["Vehicles Per Day"].mean().reindex(order).reset_index()
    colors = [COLORS["good"] if day in order[:5] else COLORS["warning"] for day in stats["Day of Week"]]
    fig = go.Figure(
        go.Bar(
            x=stats["Day of Week"],
            y=stats["Vehicles Per Day"],
            marker_color=colors,
            text=[f"{int(v):,}" for v in stats["Vehicles Per Day"]],
            textposition="outside",
            hovertemplate="%{x}<br>%{y:,.0f} vehicles<extra></extra>",
        )
    )
    fig.update_layout(
        height=360,
        margin=dict(l=10, r=10, t=20, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(17,24,39,0.74)",
        showlegend=False,
        yaxis_title="Average vehicles",
        xaxis_title="",
        font=dict(color=COLORS["text"]),
    )
    fig.update_yaxes(gridcolor="rgba(148,163,184,0.12)")
    st.plotly_chart(fig, width="stretch")


def render_monthly_patterns(historical: pd.DataFrame) -> None:
    recent_years = sorted(historical["Year"].unique())[-3:]
    monthly = (
        historical.loc[historical["Year"].isin(recent_years)]
        .groupby(["Year", "Month"])["Vehicles Per Day"]
        .mean()
        .reset_index()
    )
    fig = go.Figure()
    for year in recent_years:
        year_data = monthly.loc[monthly["Year"] == year]
        fig.add_trace(
            go.Scatter(
                x=year_data["Month"],
                y=year_data["Vehicles Per Day"],
                mode="lines+markers",
                name=str(year),
                hovertemplate=f"{year}<br>Month %{{x}}<br>%{{y:,.0f}} vehicles<extra></extra>",
            )
        )
    fig.update_layout(
        height=360,
        margin=dict(l=10, r=10, t=20, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(17,24,39,0.74)",
        yaxis_title="Average vehicles",
        xaxis_title="Month",
        xaxis=dict(
            tickmode="array",
            tickvals=list(range(1, 13)),
            ticktext=["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
        ),
        font=dict(color=COLORS["text"]),
    )
    fig.update_yaxes(gridcolor="rgba(148,163,184,0.12)")
    st.plotly_chart(fig, width="stretch")


def render_forecast_table(forecast: pd.DataFrame, historical: pd.DataFrame) -> None:
    table = forecast[["ds", "Day of Week", "Ensemble"]].copy()
    table.columns = ["Date", "Day", "Forecasted Vehicles"]
    table["Visit signal"] = table["Forecasted Vehicles"].apply(lambda v: label_for_volume(v, historical)[0])
    st.dataframe(table, hide_index=True, width="stretch")


def main() -> None:
    inject_styles()

    try:
        historical = prepare_historical()
        forecast = prepare_forecast()
    except Exception as exc:
        st.error(f"Unable to load dashboard data: {exc}")
        st.stop()

    with st.sidebar:
        st.markdown("### Trip Filters")
        day_filter = st.radio(
            "Show recommendations for",
            ["All upcoming days", "Weekdays only", "Weekends only"],
            index=0,
        )
        show_history = st.toggle("Show recent history on chart", value=True)
        st.markdown("---")
        st.caption("Use filters if your trip is limited to weekdays or weekends.")
        st.caption(
            f"Forecast window: {forecast['ds'].min().strftime('%b %d')} to {forecast['ds'].max().strftime('%b %d, %Y')}"
        )

    filtered_forecast = filter_forecast_days(forecast, day_filter)
    summary = compute_recommendations(forecast, historical, day_filter)
    if summary is None:
        st.warning("No forecast days match the selected filter.")
        st.stop()

    render_hero(summary, historical)
    render_mini_metrics(summary, filtered_forecast)

    st.markdown("### Forecast")
    st.caption("Best and worst days are marked directly on the chart. Weekend periods are shaded.")
    with st.container(border=False):
        st.markdown('<div class="section-shell">', unsafe_allow_html=True)
        render_forecast_chart(historical, filtered_forecast, summary, show_history)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("### Planning Picks")
    render_pick_pills(summary.top_days, historical, "Best days to visit", "good")
    render_pick_pills(summary.avoid_days, historical, "Days to avoid", "bad")

    st.markdown("### Quick Insights")
    render_insight_cards(summary)

    st.markdown("### Deep Dive")
    tab1, tab2, tab3 = st.tabs(["Seasonal patterns", "Weekly patterns", "Full forecast"])
    with tab1:
        render_monthly_patterns(historical)
    with tab2:
        render_day_of_week_patterns(historical)
    with tab3:
        render_forecast_table(filtered_forecast, historical)

    with st.expander("About this forecast"):
        st.markdown(
            """
            - Data source: combined two-way traffic counts for Banff townsite access.
            - Historical coverage: July 2013 to present.
            - Forecast source: ensemble predictions generated from the latest saved model outputs.
            - This dashboard is optimized for choosing quieter visit days, not forecasting exact hourly congestion.
            """
        )


if __name__ == "__main__":
    main()
