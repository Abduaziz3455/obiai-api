"""
Pydantic models for prediction validation.
"""
from datetime import datetime, date
from typing import Optional

from pydantic import BaseModel, Field


class LocationConfig(BaseModel):
    """Location configuration for weather data."""

    latitude: float = Field(..., ge=-90, le=90, description="Latitude coordinate")
    longitude: float = Field(..., ge=-180, le=180, description="Longitude coordinate")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "latitude": 40.48,
                    "longitude": 65.355
                }
            ]
        }
    }


class CropConfig(BaseModel):
    """Crop configuration for feature engineering."""

    planting_date: date = Field(..., description="Crop planting date (YYYY-MM-DD)")
    crop_type: str = Field(default="cotton", description="Crop type (e.g., cotton, wheat)")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "planting_date": "2025-04-15",
                    "crop_type": "cotton"
                }
            ]
        }
    }


class PredictionRequest(BaseModel):
    """Request model for irrigation prediction."""

    device_id: str = Field(..., min_length=1, max_length=50, description="Sensor device identifier")
    location: LocationConfig = Field(..., description="Geographic location")
    crop_config: Optional[CropConfig] = Field(default=None, description="Crop configuration (optional)")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "device_id": "sensor_001",
                    "location": {
                        "latitude": 40.48,
                        "longitude": 65.355
                    },
                    "crop_config": {
                        "planting_date": "2025-04-15",
                        "crop_type": "cotton"
                    }
                }
            ]
        }
    }


class SensorDataSummary(BaseModel):
    """Summary of sensor data used for prediction."""

    device_id: str
    timestamp: datetime
    soil_moisture: float
    soil_temperature: float


class WeatherSummary(BaseModel):
    """Summary of weather data used for prediction."""

    air_temperature: float
    precipitation_24h: float
    wind_speed: float
    solar_radiation: float


class PredictionResponse(BaseModel):
    """Response model for irrigation prediction."""

    id: int = Field(..., description="Unique prediction identifier")
    irrigation_needed: int = Field(..., description="Whether irrigation is needed (0=No, 1=Yes)")
    recommended_water_percent: float = Field(..., ge=0, le=100, description="Recommended water amount (%)")
    irrigation_time_min: float = Field(..., ge=0, le=300, description="Recommended irrigation duration (minutes)")

    # Confidence metrics (clearer naming)
    confidence_score: float = Field(..., ge=0, le=1, description="Model confidence in this prediction (0-1). Higher is better.")

    # Additional context
    sensor_data: SensorDataSummary = Field(..., description="Sensor data used")
    weather_summary: WeatherSummary = Field(..., description="Weather data used")
    timestamp: datetime = Field(..., description="Prediction timestamp")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "id": 123456789,
                    "irrigation_needed": 1,
                    "recommended_water_percent": 65.5,
                    "irrigation_time_min": 120.0,
                    "confidence_score": 0.87,
                    "sensor_data": {
                        "device_id": "sensor_001",
                        "timestamp": "2025-11-19T17:55:00Z",
                        "soil_moisture": 35.5,
                        "soil_temperature": 22.8
                    },
                    "weather_summary": {
                        "air_temperature": 28.5,
                        "precipitation_24h": 0.0,
                        "wind_speed": 5.2,
                        "solar_radiation": 650.0
                    },
                    "timestamp": "2025-11-19T18:00:00Z"
                }
            ]
        }
    }
