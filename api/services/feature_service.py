"""
Feature engineering service for irrigation prediction.
Combines sensor and weather data to generate model features.
"""
import pandas as pd
import numpy as np
from datetime import datetime, date
from typing import Dict, Optional

from src.data_processing.feature_engineer import FeatureEngineer


class FeatureService:
    """
    Service for orchestrating feature engineering pipeline.
    """

    def __init__(self):
        """Initialize feature service."""
        self.feature_engineer = FeatureEngineer()

    async def prepare_features(
        self,
        sensor_data: Dict,
        weather_df: pd.DataFrame,
        planting_date: Optional[date] = None
    ) -> pd.DataFrame:
        """
        Prepare all 42 features required by the model.

        Args:
            sensor_data: Dict with latest sensor reading {device_id, timestamp, humidity_percent, temperature}
            weather_df: DataFrame with weather data (columns: timestamp, temperature_2m, precipitation, wind_speed_10m, shortwave_radiation)
            planting_date: Crop planting date (default: April 15)

        Returns:
            DataFrame with 42 features in correct order for model

        Raises:
            ValueError: If required data is missing
        """
        if weather_df.empty:
            raise ValueError("Weather data is required")

        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(weather_df['timestamp']):
            weather_df['timestamp'] = pd.to_datetime(weather_df['timestamp'])

        # Sort by timestamp
        weather_df = weather_df.sort_values('timestamp').reset_index(drop=True)

        # Add sensor data to weather DataFrame
        # The sensor reading corresponds to current time (latest in weather data)
        weather_df['soil_temperature_0_to_7cm'] = sensor_data['temperature']
        weather_df['soil_moisture'] = sensor_data['humidity_percent'] / 100.0  # Convert to 0-1

        # Add basic time features
        df = self._add_time_features(weather_df)

        # Add rolling features
        df = self.feature_engineer.add_rolling_features(df)

        # Add derived features
        df = self.feature_engineer.add_derived_features(df)

        # Add cotton features
        if planting_date is None:
            planting_date_str = "04-15"  # Default: April 15
        else:
            planting_date_str = planting_date.strftime("%m-%d")

        df = self.feature_engineer.add_cotton_features(df, planting_date=planting_date_str)

        # Calculate etc_mm_day (ET crop = ET0 * crop coefficient)
        df['etc_mm_day'] = df['et0_estimate'] * df['crop_kc']

        # Get latest row (current prediction point)
        latest_df = df.iloc[[-1]].copy()

        # Select features in correct order (42 features from model_config.json)
        feature_cols = [
            "year", "month", "day", "hour", "day_of_year", "day_of_week",
            "season", "is_daytime", "is_growing_season",
            "temperature_2m", "precipitation", "wind_speed_10m", "shortwave_radiation",
            "soil_temperature_0_to_7cm",
            "temp_24h_mean", "temp_24h_max", "temp_24h_min", "temp_7d_mean", "temp_14d_mean",
            "temp_day_night_diff",
            "precip_24h_sum", "precip_7d_sum", "precip_14d_sum", "precip_30d_sum",
            "days_since_rain", "consecutive_dry_days",
            "radiation_24h_mean", "radiation_7d_mean", "radiation_24h_sum",
            "wind_24h_mean", "wind_7d_mean",
            "soil_temp_24h_mean", "soil_temp_7d_mean",
            "et0_estimate", "vpd_estimate", "hot_dry_days_7d",
            "days_since_planting", "crop_stage", "crop_kc", "is_critical_period",
            "gdd_cumulative",
            "soil_moisture",
            "etc_mm_day"
        ]

        # Ensure all features exist
        missing_features = [col for col in feature_cols if col not in latest_df.columns]
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")

        # Return features in correct order
        features_df = latest_df[feature_cols].copy()

        return features_df

    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add basic time-based features.

        Args:
            df: DataFrame with timestamp column

        Returns:
            DataFrame with time features added
        """
        # Basic time features
        df['year'] = df['timestamp'].dt.year
        df['month'] = df['timestamp'].dt.month
        df['day'] = df['timestamp'].dt.day
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_year'] = df['timestamp'].dt.dayofyear
        df['day_of_week'] = df['timestamp'].dt.dayofweek

        # Season (1=Winter, 2=Spring, 3=Summer, 4=Autumn for Uzbekistan)
        df['season'] = df['month'].apply(self._get_season)

        # Daytime flag (6am-8pm = daytime)
        df['is_daytime'] = ((df['hour'] >= 6) & (df['hour'] <= 20)).astype(int)

        # Growing season flag (April-September for cotton in Uzbekistan)
        df['is_growing_season'] = ((df['month'] >= 4) & (df['month'] <= 9)).astype(int)

        return df

    @staticmethod
    def _get_season(month: int) -> int:
        """
        Get season number for Uzbekistan.

        Args:
            month: Month number (1-12)

        Returns:
            Season number (1=Winter, 2=Spring, 3=Summer, 4=Autumn)
        """
        if month in [12, 1, 2]:
            return 1  # Winter
        elif month in [3, 4, 5]:
            return 2  # Spring
        elif month in [6, 7, 8]:
            return 3  # Summer
        else:
            return 4  # Autumn
