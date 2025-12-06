"""
Async weather service for fetching and caching weather data.
"""
import httpx
import asyncio
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any
import pandas as pd

from api.database.models import WeatherCache


class WeatherService:
    """
    Async service for fetching weather data from Open-Meteo API with caching.
    """

    def __init__(
        self,
        base_url: str = "https://api.open-meteo.com/v1",
        timeout: int = 30,
        max_retries: int = 3,
        cache_ttl_seconds: int = 3600
    ):
        """
        Initialize weather service.

        Args:
            base_url: Base URL for Open-Meteo API
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            cache_ttl_seconds: Cache time-to-live in seconds (default: 1 hour)
        """
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.cache_ttl_seconds = cache_ttl_seconds
        self.retry_delay = 5  # seconds

    async def get_current_and_historical_weather(
        self,
        latitude: float,
        longitude: float,
        hours_back: int = 720  # 30 days
    ) -> pd.DataFrame:
        """
        Fetch current and historical weather data.

        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate
            hours_back: Number of hours of historical data to fetch (default: 720 = 30 days)

        Returns:
            DataFrame with weather data including current and historical

        Raises:
            Exception: If API request fails
        """
        # Calculate past days for API
        past_days = min(hours_back // 24, 92)  # Open-Meteo limit is 92 days

        # Define parameters
        hourly_params = [
            "temperature_2m",
            "precipitation",
            "wind_speed_10m",
            "shortwave_radiation"
        ]

        # Check cache first
        location_key = f"{latitude}_{longitude}"
        cached_data = await self._get_cached_weather(location_key, hours_back)

        if cached_data is not None and not cached_data.empty:
            return cached_data

        # Fetch from API
        data = await self._fetch_forecast(
            latitude=latitude,
            longitude=longitude,
            hourly_params=hourly_params,
            past_days=past_days
        )

        # Convert to DataFrame
        df = self._parse_weather_response(data)

        # Cache the data
        await self._cache_weather_data(location_key, df)

        return df

    async def _fetch_forecast(
        self,
        latitude: float,
        longitude: float,
        hourly_params: List[str],
        past_days: int = 30,
        forecast_days: int = 1
    ) -> Dict[str, Any]:
        """
        Fetch weather forecast with historical data using httpx.

        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate
            hourly_params: List of hourly parameters
            past_days: Number of past days to include
            forecast_days: Number of forecast days

        Returns:
            API response as dictionary

        Raises:
            Exception: If request fails after all retries
        """
        endpoint = f"{self.base_url}/forecast"

        params = {
            "latitude": latitude,
            "longitude": longitude,
            "hourly": ",".join(hourly_params),
            "past_days": min(past_days, 92),
            "forecast_days": min(forecast_days, 16),
            "timezone": "UTC"
        }

        for attempt in range(self.max_retries):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.get(endpoint, params=params)

                    if response.status_code == 200:
                        data = response.json()

                        # Validate response
                        if self._validate_response(data):
                            return data
                        else:
                            raise ValueError("Invalid response format")

                    elif response.status_code == 429:
                        # Rate limit exceeded
                        await asyncio.sleep(self.retry_delay * 2)

                    else:
                        # Other error
                        if attempt < self.max_retries - 1:
                            await asyncio.sleep(self.retry_delay)
                        else:
                            raise Exception(f"API request failed: {response.status_code}")

            except httpx.TimeoutException:
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay)
                else:
                    raise Exception("Request timed out after all retries")

            except httpx.ConnectError as e:
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay)
                else:
                    raise Exception(f"Connection failed after all retries: {str(e)}")

            except Exception as e:
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay)
                else:
                    raise

        raise Exception("Request failed after all retries")

    def _validate_response(self, data: Dict) -> bool:
        """
        Validate API response format.

        Args:
            data: Response data

        Returns:
            True if valid, False otherwise
        """
        # Check for error
        if isinstance(data, dict) and data.get("error"):
            return False

        # Check required fields
        required_fields = ["latitude", "longitude", "hourly"]
        for field in required_fields:
            if field not in data:
                return False

        return True

    def _parse_weather_response(self, data: Dict) -> pd.DataFrame:
        """
        Parse weather API response into DataFrame.

        Args:
            data: Weather API response

        Returns:
            DataFrame with weather data
        """
        hourly_data = data.get("hourly", {})

        # Create DataFrame
        df = pd.DataFrame({
            "timestamp": pd.to_datetime(hourly_data.get("time", [])),
            "temperature_2m": hourly_data.get("temperature_2m", []),
            "precipitation": hourly_data.get("precipitation", []),
            "wind_speed_10m": hourly_data.get("wind_speed_10m", []),
            "shortwave_radiation": hourly_data.get("shortwave_radiation", [])
        })

        # Fill NaN values with 0 for safety (API sometimes returns null)
        df = df.fillna({
            "temperature_2m": 0,
            "precipitation": 0,
            "wind_speed_10m": 0,
            "shortwave_radiation": 0
        })

        return df

    async def _get_cached_weather(
        self,
        location_key: str,
        hours_back: int
    ) -> Optional[pd.DataFrame]:
        """
        Get cached weather data if available and not expired.

        Args:
            location_key: Location identifier (lat_lon)
            hours_back: Number of hours needed

        Returns:
            DataFrame with cached data or None if cache miss
        """
        try:
            # Calculate cutoff time
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours_back)

            # Query cached data
            cached_entries = await WeatherCache.filter(
                location_key=location_key,
                timestamp__gte=cutoff_time,
                expires_at__gt=datetime.now(timezone.utc)
            ).all()

            if not cached_entries or len(cached_entries) < hours_back * 0.8:
                # Not enough cached data
                return None

            # Convert to DataFrame
            data = []
            for entry in cached_entries:
                data.append({
                    "timestamp": entry.timestamp,
                    "temperature_2m": float(entry.temperature_2m) if entry.temperature_2m else None,
                    "precipitation": float(entry.precipitation) if entry.precipitation else None,
                    "wind_speed_10m": float(entry.wind_speed_10m) if entry.wind_speed_10m else None,
                    "shortwave_radiation": float(entry.shortwave_radiation) if entry.shortwave_radiation else None
                })

            df = pd.DataFrame(data)
            return df if not df.empty else None

        except Exception:
            return None

    async def _cache_weather_data(
        self,
        location_key: str,
        df: pd.DataFrame
    ) -> None:
        """
        Cache weather data in database.

        Args:
            location_key: Location identifier (lat_lon)
            df: DataFrame with weather data
        """
        try:
            expires_at = datetime.now(timezone.utc) + timedelta(seconds=self.cache_ttl_seconds)

            # Prepare data for bulk insert
            cache_entries = []
            for _, row in df.iterrows():
                cache_entries.append({
                    "location_key": location_key,
                    "timestamp": row["timestamp"],
                    "temperature_2m": row.get("temperature_2m"),
                    "precipitation": row.get("precipitation"),
                    "wind_speed_10m": row.get("wind_speed_10m"),
                    "shortwave_radiation": row.get("shortwave_radiation"),
                    "expires_at": expires_at
                })

            # Bulk create (ignore duplicates)
            for entry in cache_entries:
                try:
                    await WeatherCache.get_or_create(**entry)
                except Exception:
                    # Ignore duplicate errors
                    pass

            # Clean up expired entries
            await WeatherCache.filter(expires_at__lt=datetime.now(timezone.utc)).delete()

        except Exception as e:
            # Cache failures should not break the service
            print(f"Failed to cache weather data: {e}")
            pass
