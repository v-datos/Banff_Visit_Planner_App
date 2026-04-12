import os
import pandas as pd
import numpy as np
from utilsforecast.feature_engineering import fourier, trend, pipeline, partial
from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATSx, NHITS
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    # Load latest traffic data up to the last available date
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'TW Traffic _data.csv')
    logger.info(f"Loading data from {data_path}")
    
    traffic_df = pd.read_csv(data_path, encoding='utf-16', sep='\t')
    
    # Rename columns to standard names used by the dashboard and previous script
    traffic_df.rename(columns={'Day of Time Stamp': 'Date', 'Combined TW': 'Vehicles Per Day'}, inplace=True)
    
    # Drop the 'Grand Total' dummy row if it exists (usually the last row)
    traffic_df = traffic_df[traffic_df['Date'] != 'Grand Total']
    
    # Change 'Date' to datetime
    traffic_df['Date'] = pd.to_datetime(traffic_df['Date'], errors='coerce')
    
    # Create Vehicles per Day dataset
    vpd_df = traffic_df[['Date', 'Vehicles Per Day']].copy()
    vpd_df['Vehicles Per Day'] = pd.to_numeric(vpd_df['Vehicles Per Day'], errors='coerce')
    vpd_df = vpd_df.dropna(subset=['Date', 'Vehicles Per Day'])

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

    exo_variables = [col for col in df_train_exo.columns if col not in ['unique_id', 'ds', 'y']]

    logger.info("Loading pretrained models...")
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'modelos_exo')
    nf_nbeatsx = NeuralForecast.load(path=model_dir)

    logger.info("Generating predictions...")
    # Forecast next 15 days
    df_pred = nf_nbeatsx.predict(df=df_train_exo, futr_df=df_train_exo_fut)

    # Select best results and Ensemble
    model_names = [col for col in df_pred.columns if col not in ['unique_id', 'ds']]
    df_pred['Ensemble'] = df_pred[model_names].mean(axis=1)

    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'predictions.csv')
    df_pred.to_csv(output_path, index=False)
    logger.info(f"Predictions saved successfully to {output_path}")

if __name__ == "__main__":
    main()
