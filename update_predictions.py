import os
import importlib.metadata as metadata
import pandas as pd
import numpy as np
from pathlib import Path
from utilsforecast.feature_engineering import fourier, trend, pipeline, partial
from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATSx, NHITS
import logging
from data_utils import load_csv, normalize_historical_traffic

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def log_runtime_versions() -> None:
    for package in [
        "python",
        "pandas",
        "torch",
        "pytorch-lightning",
        "neuralforecast",
        "utilsforecast",
    ]:
        if package == "python":
            version = os.sys.version.replace("\n", " ")
        else:
            version = metadata.version(package)
        logger.info("%s version: %s", package, version)

def prepare_data_for_model(vpd_df, date_col='Date', target_col='Vehicles Per Day'):
    """Prepare data in MLForecast format with comprehensive exploration"""
    logger.info("Preparing MLForecast data from scratch...")
    logger.info(f"Input shape: {vpd_df.shape}")

    # Create MLForecast format
    ml_df = pd.DataFrame({
        'unique_id': 'traffic',
        'ds': pd.to_datetime(vpd_df[date_col]),
        'y': vpd_df[target_col].astype(float)
    })

    # Sort by date
    ml_df = ml_df.sort_values('ds').reset_index(drop=True)

    logger.info(f"MLForecast data prepared: {ml_df.shape}")
    logger.info(f"Date range: {ml_df['ds'].min()} to {ml_df['ds'].max()}")
    logger.info(f"Target statistics: min={ml_df['y'].min():.0f}, max={ml_df['y'].max():.0f}, mean={ml_df['y'].mean():.0f}")

    return ml_df

def main():
    log_runtime_versions()

    # Load latest traffic data up to the last available date
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'TW Traffic _data.csv')
    logger.info(f"Loading data from {data_path}")

    traffic_df = load_csv(Path(data_path), encoding='utf-16', sep='\t')
    traffic_df = normalize_historical_traffic(traffic_df)
    logger.info(
        "Normalized historical data: %s rows from %s to %s",
        len(traffic_df),
        traffic_df['Date'].min(),
        traffic_df['Date'].max(),
    )

    # Create Vehicles per Day dataset
    vpd_df = traffic_df[['Date', 'Vehicles Per Day']].copy()

    # Prepare data for model
    ml_df = prepare_data_for_model(vpd_df)

    # Train on all available data (dynamic, not hardcoded to Sept 2025)
    df_train = ml_df.copy()
    
    horizon = 15

    # Create exogenous features
    features = [
        trend,
        partial(fourier, season_length=7, k=3),
        partial(fourier, season_length=365, k=3),
        partial(fourier, season_length=730, k=3),
    ]

    transformed_df, future_df = pipeline(
        df_train,
        features=features,
        freq='D',
        h=horizon,
    )

    # Prepare data for NF
    df_train_exo = transformed_df.copy()
    # Create seasonal time features
    df_train_exo['weekday'] = df_train_exo['ds'].dt.dayofweek
    df_train_exo['month'] = df_train_exo['ds'].dt.month
    df_train_exo['year'] = df_train_exo['ds'].dt.year

    df_train_exo_fut = future_df.copy()
    # Create seasonal time features
    df_train_exo_fut['weekday'] = df_train_exo_fut['ds'].dt.dayofweek
    df_train_exo_fut['month'] = df_train_exo_fut['ds'].dt.month
    df_train_exo_fut['year'] = df_train_exo_fut['ds'].dt.year

    logger.info("Loading pretrained models...")
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'modelos_exo')
    nf_nbeatsx = NeuralForecast.load(path=model_dir, map_location="cpu")

    logger.info("Generating predictions...")
    # Forecast next 15 days
    df_pred = nf_nbeatsx.predict(df=df_train_exo, futr_df=df_train_exo_fut)

    # Select best results and Ensemble
    model_names = [col for col in df_pred.columns if col not in ['unique_id', 'ds']]
    df_pred['Ensemble'] = df_pred[model_names].mean(axis=1)

    recent_history = vpd_df['Vehicles Per Day'].tail(30)
    recent_median = recent_history.median()
    recent_min = recent_history.min()
    recent_max = recent_history.max()

    if df_pred['Ensemble'].isna().any():
        raise ValueError("Forecast contains NaN values")

    if (df_pred['Ensemble'] < recent_median * 0.5).all():
        raise ValueError(
            "Forecast is implausibly low: "
            f"median={df_pred['Ensemble'].median():.0f}, "
            f"recent history median={recent_median:.0f}, "
            f"range=({recent_min:.0f}, {recent_max:.0f})"
        )

    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'predictions.csv')
    df_pred.to_csv(output_path, index=False)
    logger.info(f"Predictions saved successfully to {output_path}")

if __name__ == "__main__":
    main()
