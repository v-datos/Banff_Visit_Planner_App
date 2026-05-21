<p align="center">
  <img src="visit_planner.png" alt="Banff Visit Planner Header" width="800">
</p>


# Banff Visit Planner

[Open the live app](https://banff-visit-planner-app.streamlit.app/)

Banff Visit Planner is a public trip-planning app that helps visitors choose quieter days to visit Banff by turning traffic data into simple recommendations.

Instead of making people interpret raw charts, the app answers a practical question fast:

**When should I go to Banff to avoid the heaviest traffic?**

## What the app does

- Highlights the best upcoming day to visit
- Flags the busiest day to avoid
- Shows a short-term traffic forecast for the next 15 days
- Surfaces a few recommended visit options and days to avoid
- Gives extra context with weekly and seasonal traffic patterns

## How it works

The app combines two inputs:

1. Historical Banff traffic data from the Town of Banff's public Tableau dashboard
2. A forecasting pipeline that generates updated short-term traffic predictions

Those predictions are published into the app so visitors can quickly compare upcoming days and choose a lower-traffic visit window.

The public app uses a lightweight runtime dependency set for deployment, while the scheduled data-refresh pipeline installs additional forecasting and extraction packages separately.

## Automatic updates

This repository includes an automated update pipeline.

- Traffic data is refreshed from the public source
- Forecasts are regenerated
- The updated files are committed back to GitHub
- Streamlit Community Cloud redeploys from the repo, which keeps the live app current

The update workflow is set to run on the 1st and 16th of each month.

## Who this is for

- Tourists planning a Banff trip
- Casual visitors choosing between weekdays and weekends
- Anyone who wants a quick, non-technical recommendation about expected traffic

## Live app

- Public app: [banff-visit-planner-app.streamlit.app](https://banff-visit-planner-app.streamlit.app/)

## Repository guide

- Main app: [traffic_app.py](/Users/micra/Dataland/visit_planner_app/traffic_app.py:1)
- Technical documentation: [documentation.md](/Users/micra/Dataland/visit_planner_app/documentation.md:1)
- Forecast updater: [update_predictions.py](/Users/micra/Dataland/visit_planner_app/update_predictions.py:1)
- Data extractor: [extract_data.py](/Users/micra/Dataland/visit_planner_app/extract_data.py:1)

## Local run

```bash
streamlit run traffic_app.py
```

## Data source

- [Banff Traffic Data Dashboard](https://public.tableau.com/app/profile/banff.gis/viz/BanffTrafficData-GS/TWTraffic)
