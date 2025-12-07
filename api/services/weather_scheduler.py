"""
Weather data scheduler service.
Fetches weather data from Open-Meteo API and stores it in the database every 5 minutes.
"""
import asyncio
import httpx
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any
import pandas as pd

from api.database.models import WeatherData


class WeatherScheduler:
    """
    Scheduler service for fetching and storing weather data periodically.
    """

    def __init__(
        self,
        base_url: str = "https://api.open-meteo.com/v1",
        timeout: int = 30,
        max_retries: int = 3,
        fetch_interval_seconds: int = 300  # 5 minutes
    ):
        """
        Initialize weather scheduler.

        Args:
            base_url: Base URL for Open-Meteo API
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            fetch_interval_seconds: Interval between fetches in seconds (default: 300 = 5 minutes)
        """
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.fetch_interval_seconds = fetch_interval_seconds
        self.retry_delay = 5  # seconds
        self.running = False
        self.task = None

        # Default locations to fetch weather for (can be configured)
        self.locations = []

    def add_location(self, latitude: float, longitude: float):
        """
        Add a location to fetch weather data for.

        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate
        """
        location_key = f"{latitude}_{longitude}"
        if location_key not in [loc['key'] for loc in self.locations]:
            self.locations.append({
                'key': location_key,
                'latitude': latitude,
                'longitude': longitude
            })

    def remove_location(self, latitude: float, longitude: float):
        """
        Remove a location from the fetch list.

        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate
        """
        location_key = f"{latitude}_{longitude}"
        self.locations = [loc for loc in self.locations if loc['key'] != location_key]

    async def fetch_and_store_weather(self, latitude: float, longitude: float):
        """
        Fetch current and historical weather data and store in database.

        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate
        """
        try:
            # Fetch weather data
            weather_data = await self._fetch_forecast(
                latitude=latitude,
                longitude=longitude,
                hourly_params=["temperature_2m", "precipitation", "wind_speed_10m", "shortwave_radiation"],
                past_days=30,  # Get 30 days of historical data
                forecast_days=1
            )

            # Parse response
            df = self._parse_weather_response(weather_data)

            if df.empty:
                print(f"No weather data fetched for {latitude}, {longitude}")
                return

            # Store in database
            location_key = f"{latitude}_{longitude}"
            await self._store_weather_data(location_key, latitude, longitude, df)

            print(f"✓ Weather data stored for {location_key} ({len(df)} records)")

        except Exception as e:
            print(f"✗ Failed to fetch/store weather for {latitude}, {longitude}: {str(e)}")

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

        # Create DataFrame with timezone-aware timestamps
        df = pd.DataFrame({
            "timestamp": pd.to_datetime(hourly_data.get("time", []), utc=True),
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

    async def _store_weather_data(
        self,
        location_key: str,
        latitude: float,
        longitude: float,
        df: pd.DataFrame
    ) -> None:
        """
        Store weather data in database.

        Args:
            location_key: Location identifier (lat_lon)
            latitude: Latitude coordinate
            longitude: Longitude coordinate
            df: DataFrame with weather data
        """
        try:
            # Prepare data for bulk insert
            for _, row in df.iterrows():
                try:
                    # Use get_or_create to avoid duplicates
                    await WeatherData.update_or_create(
                        location_key=location_key,
                        timestamp=row["timestamp"],
                        defaults={
                            "latitude": latitude,
                            "longitude": longitude,
                            "temperature_2m": row.get("temperature_2m"),
                            "precipitation": row.get("precipitation"),
                            "wind_speed_10m": row.get("wind_speed_10m"),
                            "shortwave_radiation": row.get("shortwave_radiation"),
                        }
                    )
                except Exception as e:
                    # Ignore individual insert errors
                    pass

            # Clean up old data (keep last 32 days only)
            cutoff_time = datetime.now(timezone.utc) - timedelta(days=32)
            await WeatherData.filter(
                location_key=location_key,
                timestamp__lt=cutoff_time
            ).delete()

        except Exception as e:
            # Storage failures should not break the service
            print(f"Failed to store weather data: {e}")
            pass

    async def _scheduler_loop(self):
        """
        Main scheduler loop that runs continuously.
        """
        print(f"🌤️  Weather scheduler started (interval: {self.fetch_interval_seconds}s)")

        while self.running:
            try:
                if not self.locations:
                    print("⚠️  No locations configured for weather fetching")
                else:
                    # Fetch weather for all configured locations
                    tasks = [
                        self.fetch_and_store_weather(loc['latitude'], loc['longitude'])
                        for loc in self.locations
                    ]
                    await asyncio.gather(*tasks, return_exceptions=True)

                # Wait for next interval
                await asyncio.sleep(self.fetch_interval_seconds)

            except asyncio.CancelledError:
                print("🛑 Weather scheduler cancelled")
                break
            except Exception as e:
                print(f"✗ Scheduler loop error: {str(e)}")
                # Continue running despite errors
                await asyncio.sleep(self.fetch_interval_seconds)

    async def start(self):
        """
        Start the weather scheduler.
        """
        if self.running:
            print("Weather scheduler is already running")
            return

        self.running = True
        self.task = asyncio.create_task(self._scheduler_loop())

    async def stop(self):
        """
        Stop the weather scheduler.
        """
        if not self.running:
            print("Weather scheduler is not running")
            return

        self.running = False
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
        print("🛑 Weather scheduler stopped")


# Global scheduler instance
weather_scheduler = WeatherScheduler()
