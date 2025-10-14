import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

st.set_page_config(page_title="Plan Your Banff Visit: AI-Powered Traffic Insights", layout="wide")

DATA_DIR = Path(__file__).resolve().parent
HISTORICAL_PATH = DATA_DIR / "TW Traffic _data.csv"
FORECAST_PATH = DATA_DIR / "predictions.csv"


@st.cache_data(show_spinner=False)
def load_csv(path: Path, **read_kwargs) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path.name} not found in {path.parent}")
    return pd.read_csv(path, **read_kwargs)


def prepare_historical() -> pd.DataFrame:
    """Load and prepare historical traffic data"""
    df = load_csv(HISTORICAL_PATH, encoding="utf-16", sep="\t")
    df.columns = [col.strip() for col in df.columns]
    rename_map = {"Day of Time Stamp": "Date", "Combined TW": "Vehicles Per Day"}
    missing = [src for src in rename_map if src not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in historical data: {', '.join(missing)}")
    df = df.rename(columns=rename_map)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Vehicles Per Day"] = pd.to_numeric(df["Vehicles Per Day"], errors="coerce")
    df = df.dropna(subset=["Date", "Vehicles Per Day"])
    df = df[:-1]
    df = df.sort_values("Date")
    df = df[~df["Date"].duplicated(keep="first")]
    
    # Add derived columns for analysis
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Month Name"] = df["Date"].dt.strftime("%B")
    df["Day of Week"] = df["Date"].dt.day_name()
    df["Day of Week Num"] = df["Date"].dt.dayofweek
    df["Week"] = df["Date"].dt.isocalendar().week
    df["Is Weekend"] = df["Day of Week Num"].isin([5, 6])
    
    return df


def prepare_forecast() -> pd.DataFrame:
    """Load and prepare forecast data"""
    df = load_csv(FORECAST_PATH)
    df.columns = [col.strip() for col in df.columns]
    required_cols = {"ds", "Ensemble"}
    if not required_cols.issubset(df.columns):
        missing = required_cols.difference(df.columns)
        raise KeyError(f"Missing columns in forecast data: {', '.join(sorted(missing))}")
    df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
    df["Ensemble"] = pd.to_numeric(df["Ensemble"], errors="coerce")
    df = df.dropna(subset=["ds", "Ensemble"])
    df = df.sort_values("ds")
    
    # Add derived columns
    df["Day of Week"] = df["ds"].dt.day_name()
    df["Day of Week Num"] = df["ds"].dt.dayofweek
    df["Is Weekend"] = df["Day of Week Num"].isin([5, 6])
    
    for col in ["yhat_lower", "yhat_upper"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def calculate_traffic_score(volume: float, avg_volume: float, std_volume: float) -> float:
    """
    Calculate a traffic comfort score (0-100)
    Higher score = better time to visit (lower traffic)
    """
    # Normalize the volume relative to average
    z_score = (volume - avg_volume) / std_volume if std_volume > 0 else 0
    
    # Convert to 0-100 scale (inverted so lower traffic = higher score)
    # z_score of -2 (very low traffic) → 100
    # z_score of 0 (average) → 50
    # z_score of +2 (very high traffic) → 0
    score = 50 - (z_score * 25)
    score = max(0, min(100, score))  # Clamp between 0-100
    
    return round(score, 1)


def get_traffic_category(volume: float) -> tuple:
    """Return traffic category and color"""
    if volume < 20000:
        return "Light", "#2ecc71"  # Green
    elif volume < 25000:
        return "Moderate", "#f39c12"  # Orange
    elif volume < 30000:
        return "Heavy", "#e74c3c"  # Red
    else:
        return "Very Heavy", "#8b0000"  # Dark Red


def render_kpi_cards(historical: pd.DataFrame, forecast: pd.DataFrame) -> None:
    """Render KPI cards at the top of the dashboard"""
    # Current month trend (last 30 days vs previous 30)
    last_30 = historical.tail(30)["Vehicles Per Day"].mean()
    prev_30 = historical.iloc[-60:-30]["Vehicles Per Day"].mean()
    trend_pct = ((last_30 - prev_30) / prev_30 * 100) if prev_30 > 0 else 0

    # Forecast peak
    if not forecast.empty:
        fcst_peak = forecast.loc[forecast["Ensemble"].idxmax()]
        fcst_low = forecast.loc[forecast["Ensemble"].idxmin()]

    if not forecast.empty:
        st.markdown("### 🔮 Forecast Highlights")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Forecast Peak Day",
                fcst_peak["ds"].strftime("%b %d, %Y"),
                f"{int(fcst_peak['Ensemble']):,} vehicles"
            )

        with col2:
            st.metric(
                "Best Forecast Day",
                fcst_low["ds"].strftime("%b %d, %Y"),
                f"{int(fcst_low['Ensemble']):,} vehicles"
            )

        with col3:
            days_over_25k = (forecast["Ensemble"] > 25000).sum()
            st.metric(
                "High Traffic Days",
                f"{days_over_25k}/{len(forecast)}",
                "days > 25K vehicles"
            )

        with col4:
            st.metric(
                "Recent Trend (30 days)",
                f"{trend_pct:+.1f}%",
                "vs previous 30 days",
                delta_color="inverse"
            )


def render_main_chart(historical: pd.DataFrame, forecast: pd.DataFrame) -> None:
    """Render main time series chart with historical and forecast"""
    st.markdown("### 📈 Traffic Trends & Forecast")
    
    # Get last 90 days of historical data
    history_tail = historical[historical["Date"] >= forecast["ds"].min() - pd.Timedelta(days=90)]
    
    fig = go.Figure()
    
    # Historical data
    fig.add_trace(go.Scatter(
        x=history_tail["Date"],
        y=history_tail["Vehicles Per Day"],
        mode='lines',
        name='Historical',
        line=dict(color='rgba(52, 152, 219, 0.5)', width=2),
        hovertemplate='<b>Date:</b> %{x|%b %d, %Y}<br><b>Vehicles:</b> %{y:,.0f}<extra></extra>'
    ))
    
    # Forecast
    fig.add_trace(go.Scatter(
        x=forecast["ds"],
        y=forecast["Ensemble"],
        mode='lines',
        name='Forecast',
        line=dict(color='#e74c3c', width=2, dash='dash'),
        hovertemplate='<b>Date:</b> %{x|%b %d, %Y}<br><b>Forecast:</b> %{y:,.0f}<extra></extra>'
    ))
    
    # Confidence interval if available
    if {"yhat_lower", "yhat_upper"}.issubset(forecast.columns):
        fig.add_trace(go.Scatter(
            x=forecast["ds"],
            y=forecast["yhat_upper"],
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ))
        fig.add_trace(go.Scatter(
            x=forecast["ds"],
            y=forecast["yhat_lower"],
            mode='lines',
            line=dict(width=0),
            fillcolor='rgba(231, 76, 60, 0.2)',
            fill='tonexty',
            name='Confidence Interval',
            hoverinfo='skip'
        ))
    
    # Add traffic level reference lines
    fig.add_hline(y=25000, line_dash="dot", line_color="orange",
                  annotation_text="Heavy Traffic (25K)", annotation_position="top left")
    fig.add_hline(y=20000, line_dash="dot", line_color="green",
                  annotation_text="Moderate Traffic (20K)", annotation_position="top left")
    
    fig.update_layout(
        title="Historical and Forecasted Daily Vehicle Traffic",
        xaxis_title="Date",
        yaxis_title="Vehicles Per Day",
        hovermode='x unified',
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_day_of_week_analysis(historical: pd.DataFrame) -> None:
    """Render day of week pattern analysis"""
    st.markdown("### 📅 Day of Week Patterns")
    
    # Calculate averages by day of week
    dow_stats = historical.groupby("Day of Week")["Vehicles Per Day"].agg(['mean', 'std', 'count'])
    
    # Sort by day of week
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    dow_stats = dow_stats.reindex(day_order)
    
    # Create bar chart
    fig = go.Figure()
    
    colors = ['#3498db' if day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'] 
              else '#e74c3c' for day in dow_stats.index]
    
    fig.add_trace(go.Bar(
        x=dow_stats.index,
        y=dow_stats['mean'],
        marker_color=colors,
        text=[f"{int(val):,}" for val in dow_stats['mean']],
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Ensemble: %{y:,.0f} vehicles<extra></extra>'
    ))
    
    fig.update_layout(
        title="Average Traffic by Day of Week",
        xaxis_title="Day of Week",
        yaxis_title="Average Vehicles Per Day",
        height=400,
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Add insights
    best_day = dow_stats['mean'].idxmin()
    worst_day = dow_stats['mean'].idxmax()
    difference = dow_stats['mean'][worst_day] - dow_stats['mean'][best_day]
    pct_diff = (difference / dow_stats['mean'][best_day]) * 100
    
    st.info(f"💡 **Insight:** {best_day}s have {pct_diff:.0f}% less traffic than {worst_day}s "
            f"({int(difference):,} fewer vehicles on average). Consider visiting mid-week!")


def render_visit_planner(forecast: pd.DataFrame, historical: pd.DataFrame) -> None:
    """Render the visit planning recommendation tool"""
    st.markdown("### 🎯 Visit Planner: Find the Best Days")
    st.caption("**Traffic Score (0-100):** 🟢 70-100 Excellent | 🟡 50-69 Good | 🟠 30-49 Fair | 🔴 0-29 Poor")
    
    # Calculate statistics for scoring
    hist_avg = historical["Vehicles Per Day"].mean()
    hist_std = historical["Vehicles Per Day"].std()
    
    # Score each forecast day
    forecast_scored = forecast.copy()
    forecast_scored["Traffic Score"] = forecast_scored["Ensemble"].apply(
        lambda x: calculate_traffic_score(x, hist_avg, hist_std)
    )
    forecast_scored["Category"], forecast_scored["Color"] = zip(
        *forecast_scored["Ensemble"].apply(get_traffic_category)
    )
    
    # Get best and worst days by score
    forecast_sorted_by_score = forecast_scored.sort_values("Traffic Score", ascending=False)

    # Display top recommendations
    st.markdown("#### 🌟 Top 5 Recommended Days to Visit")

    # Get top 5 by score, then sort chronologically
    top_5 = forecast_sorted_by_score.head(5).sort_values("ds")

    for idx, row in top_5.iterrows():
        col1, col2, col3, col4 = st.columns([2, 2, 2, 1])

        with col1:
            st.markdown(f"**{row['ds'].strftime('%A, %B %d')}**")

        with col2:
            st.markdown(f"Traffic: :green[**{row['Category']}**]")

        with col3:
            st.markdown(f"{int(row['Ensemble']):,} vehicles")

        with col4:
            score_color = "green" if row["Traffic Score"] >= 70 else "orange" if row["Traffic Score"] >= 50 else "red"
            st.markdown(f":{score_color}[**Score: {row['Traffic Score']:.0f}**]")

    st.markdown("---")

    # Display days to avoid
    st.markdown("#### ⚠️ Days to Avoid (High Traffic Expected)")

    # Get bottom 5 by score, then sort chronologically
    bottom_5 = forecast_sorted_by_score.tail(5).sort_values("ds")
    
    for idx, row in bottom_5.iterrows():
        col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
        
        with col1:
            st.markdown(f"**{row['ds'].strftime('%A, %B %d')}**")
        
        with col2:
            st.markdown(f"Traffic: :red[**{row['Category']}**]")
        
        with col3:
            st.markdown(f"{int(row['Ensemble']):,} vehicles")
        
        with col4:
            st.markdown(f":red[**Score: {row['Traffic Score']:.0f}**]")
    
    st.markdown("---")

    # Explanation of scoring
    with st.expander("ℹ️ How is the Traffic Score calculated?"):
        st.markdown("""
        The **Traffic Score** (0-100) helps you identify the best days to visit:
        
        - **Score 70-100** 🟢 Excellent - Significantly below average traffic
        - **Score 50-69** 🟡 Good - Below to average traffic
        - **Score 30-49** 🟠 Fair - Above average traffic
        - **Score 0-29** 🔴 Poor - Significantly above average traffic
        
        The score is calculated based on:
        - Historical traffic patterns (how this compares to typical days)
        - Day of week trends (weekdays generally score higher)
        - Seasonal patterns
        
        **Traffic Categories:**
        - **Light**: < 20,000 vehicles/day
        - **Moderate**: 20,000 - 25,000 vehicles/day
        - **Heavy**: 25,000 - 30,000 vehicles/day
        - **Very Heavy**: > 30,000 vehicles/day
        """)


def render_monthly_trends(historical: pd.DataFrame) -> None:
    """Render monthly comparison chart"""
    st.markdown("### 📆 Monthly Traffic Patterns")
    
    # Get recent years for comparison
    recent_years = historical["Year"].unique()[-3:]
    monthly_data = historical[historical["Year"].isin(recent_years)]
    
    # Calculate monthly averages
    monthly_avg = monthly_data.groupby(["Year", "Month"])["Vehicles Per Day"].mean().reset_index()
    
    fig = go.Figure()
    
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    for year in sorted(recent_years):
        year_data = monthly_avg[monthly_avg["Year"] == year]
        fig.add_trace(go.Scatter(
            x=year_data["Month"],
            y=year_data["Vehicles Per Day"],
            mode='lines+markers',
            name=str(year),
            hovertemplate='<b>%{fullData.name}</b><br>Month: %{x}<br>Ensemble: %{y:,.0f}<extra></extra>'
        ))
    
    fig.update_layout(
        title="Average Monthly Traffic Comparison",
        xaxis_title="Month",
        yaxis_title="Average Vehicles Per Day",
        xaxis=dict(tickmode='array', tickvals=list(range(1, 13)), ticktext=month_names),
        height=400,
        hovermode='x unified'
    )

    st.plotly_chart(fig, use_container_width=True)

    # Add insights
    monthly_overall = monthly_data.groupby("Month")["Vehicles Per Day"].mean()
    peak_month = monthly_overall.idxmax()
    lowest_month = monthly_overall.idxmin()
    peak_month_name = month_names[peak_month - 1]
    lowest_month_name = month_names[lowest_month - 1]

    # Calculate summer vs winter average
    summer_months = [6, 7, 8]  # June, July, August
    winter_months = [12, 1, 2]  # December, January, February
    summer_avg = monthly_overall[monthly_overall.index.isin(summer_months)].mean()
    winter_avg = monthly_overall[monthly_overall.index.isin(winter_months)].mean()
    summer_winter_diff = ((summer_avg - winter_avg) / winter_avg) * 100

    st.info(f"💡 **Insight:** Traffic peaks in **{peak_month_name}** and is lowest in **{lowest_month_name}**. "
            f"Summer months (Jun-Aug) see **{summer_winter_diff:.0f}% more traffic** than winter months (Dec-Feb). "
            f"Consider visiting in shoulder seasons (Apr-May or Sep-Nov) for a balance of good weather and lower crowds!")


def main() -> None:
    # Header
    st.title("🏔️ Plan Your Banff Visit: AI-Powered Traffic Insights")
    st.markdown("*Analyze historical traffic patterns and find the best days to visit Banff with lower congestion*")
    st.markdown("---")
    
    # Load data
    try:
        historical_df = prepare_historical()
        forecast_df = prepare_forecast()
    except Exception as exc:
        st.error(f"❌ Unable to load data: {exc}")
        st.stop()
    
    # Sidebar controls
    with st.sidebar:
        st.header("⚙️ Controls")
        st.markdown("---")

        # Display forecast date range (static)
        if not forecast_df.empty:
            st.subheader("Forecast Range")
            start_date = forecast_df["ds"].min().date()
            end_date = forecast_df["ds"].max().date()
            st.write(f"**{start_date.strftime('%b %d, %Y')}** to **{end_date.strftime('%b %d, %Y')}**")
            st.metric("Forecast Days", len(forecast_df))
            st.markdown("---")

        filtered_forecast = forecast_df

        st.caption("💡 **Tip:** Lower traffic scores indicate better times to visit!")

        st.markdown("---")
        st.subheader("📍 Data Source")
        st.caption("Data source: Combined two-way traffic in the Town of Banff, Alberta. Data gathered from July 2013 until the present.")
        st.caption("https://public.tableau.com/app/profile/banff.gis/viz/BanffTrafficData-GS/TWTraffic")
    
    # Main content
    render_kpi_cards(historical_df, filtered_forecast)
    st.markdown("---")

    render_visit_planner(filtered_forecast, historical_df)
    st.markdown("---")

    render_main_chart(historical_df, filtered_forecast)
    st.markdown("---")

    # Two columns for analysis
    col1, col2 = st.columns(2)

    with col1:
        render_day_of_week_analysis(historical_df)

    with col2:
        render_monthly_trends(historical_df)

    st.markdown("---")
    
    # Detailed forecast table
    with st.expander("📋 View Detailed Forecast Data"):
        display_df = filtered_forecast[["ds", "Ensemble", "Day of Week"]].copy()
        display_df.columns = ["Date", "Forecasted Vehicles", "Day of Week"]
        display_df["Traffic Category"] = display_df["Forecasted Vehicles"].apply(
            lambda x: get_traffic_category(x)[0]
        )
        st.dataframe(
            display_df.set_index("Date"),
            use_container_width=True
        )
    
    # Footer
    st.markdown("---")
    st.caption("""
    📌 **Data Source:** Combined two-way traffic at main entrances to Banff townsite  
    📅 **Historical Data:** July 2013 - Present  
    🔮 **Forecast Method:** Neural ensemble forecasting (NBEATSx + NHITSx models)  
    ⚠️ **Note:** Forecasts are estimates based on historical patterns and may vary with weather and special events
    """)


if __name__ == "__main__":
    main()