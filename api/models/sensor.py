"""
Pydantic models for sensor data validation.
"""
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class SensorDataRequest(BaseModel):
    """Request model for storing sensor data."""

    device_id: str = Field(..., min_length=1, max_length=50, description="Unique sensor device identifier")
    timestamp: datetime = Field(..., description="Timestamp of the sensor reading (ISO 8601 format)")
    humidity_raw: int = Field(..., ge=0, description="Raw humidity sensor value")
    humidity_percent: float = Field(..., ge=0, le=100, description="Soil moisture percentage (0-100)")
    temperature: float = Field(..., ge=-20, le=60, description="Soil temperature in Celsius")

    @field_validator('timestamp')
    @classmethod
    def timestamp_not_future(cls, v):
        """Validate that timestamp is not in the future."""
        if v > datetime.now(v.tzinfo):
            raise ValueError('timestamp cannot be in the future')
        return v

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "device_id": "sensor_001",
                    "timestamp": "2025-11-19T17:55:00Z",
                    "humidity_raw": 550,
                    "humidity_percent": 35.5,
                    "temperature": 22.8
                }
            ]
        }
    }


class SensorDataResponse(BaseModel):
    """Response model for stored sensor data."""

    id: int = Field(..., description="Database ID of the sensor reading")
    device_id: str = Field(..., description="Sensor device identifier")
    timestamp: datetime = Field(..., description="Timestamp of the reading")
    humidity_percent: float = Field(..., description="Soil moisture percentage")
    temperature: float = Field(..., description="Soil temperature in Celsius")
    message: str = Field(default="Sensor data stored successfully", description="Success message")

    model_config = {
        "from_attributes": True
    }


class SensorDataListResponse(BaseModel):
    """Response model for listing sensor data."""

    total: int = Field(..., description="Total number of readings")
    data: list[SensorDataResponse] = Field(..., description="List of sensor readings")
