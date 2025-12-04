"""
Pydantic models for weather data.
"""
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class WeatherData(BaseModel):
    """Weather data model."""

    timestamp: datetime
    temperature_2m: Optional[float] = Field(None, description="Air temperature at 2m (°C)")
    precipitation: Optional[float] = Field(None, description="Precipitation (mm)")
    wind_speed_10m: Optional[float] = Field(None, description="Wind speed at 10m (m/s)")
    shortwave_radiation: Optional[float] = Field(None, description="Solar radiation (W/m²)")


class WeatherCacheResponse(BaseModel):
    """Response model for weather cache."""

    location_key: str
    data_points: int
    cached: bool
    fetched_at: datetime
