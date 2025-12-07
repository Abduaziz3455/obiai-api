"""
Weather database service for retrieving stored weather data.
"""
from datetime import datetime, timedelta, timezone
from typing import Optional
import pandas as pd

from api.database.models import WeatherData


class WeatherDBService:
    """
    Service for retrieving weather data from the database.
    """

    async def get_current_and_historical_weather(
        self,
        latitude: float,
        longitude: float,
        hours_back: int = 720  # 30 days
    ) -> pd.DataFrame:
        """
        Get current and historical weather data from database.

        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate
            hours_back: Number of hours of historical data to retrieve (default: 720 = 30 days)

        Returns:
            DataFrame with weather data including current and historical

        Raises:
            Exception: If no data found or query fails
        """
        location_key = f"{latitude}_{longitude}"

        # Calculate cutoff time
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours_back)

        try:
            # Query weather data from database
            weather_records = await WeatherData.filter(
                location_key=location_key,
                timestamp__gte=cutoff_time
            ).order_by('timestamp').all()

            if not weather_records:
                raise Exception(f"No weather data found for location {latitude}, {longitude}")

            # Convert to DataFrame
            data = []
            for record in weather_records:
                data.append({
                    "timestamp": record.timestamp,
                    "temperature_2m": float(record.temperature_2m) if record.temperature_2m else 0.0,
                    "precipitation": float(record.precipitation) if record.precipitation else 0.0,
                    "wind_speed_10m": float(record.wind_speed_10m) if record.wind_speed_10m else 0.0,
                    "shortwave_radiation": float(record.shortwave_radiation) if record.shortwave_radiation else 0.0
                })

            df = pd.DataFrame(data)

            if df.empty:
                raise Exception(f"No weather data available for location {latitude}, {longitude}")

            return df

        except Exception as e:
            raise Exception(f"Failed to retrieve weather data from database: {str(e)}")

    async def get_latest_weather(
        self,
        latitude: float,
        longitude: float
    ) -> Optional[dict]:
        """
        Get the latest weather data for a location.

        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate

        Returns:
            Dictionary with latest weather data or None if not found
        """
        location_key = f"{latitude}_{longitude}"

        try:
            # Get the most recent weather record
            latest_record = await WeatherData.filter(
                location_key=location_key
            ).order_by('-timestamp').first()

            if not latest_record:
                return None

            return {
                "timestamp": latest_record.timestamp,
                "temperature_2m": float(latest_record.temperature_2m) if latest_record.temperature_2m else None,
                "precipitation": float(latest_record.precipitation) if latest_record.precipitation else None,
                "wind_speed_10m": float(latest_record.wind_speed_10m) if latest_record.wind_speed_10m else None,
                "shortwave_radiation": float(latest_record.shortwave_radiation) if latest_record.shortwave_radiation else None
            }

        except Exception:
            return None
