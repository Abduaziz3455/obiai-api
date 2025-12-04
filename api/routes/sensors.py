"""
Sensor data endpoints for storing IoT sensor readings.
"""
from fastapi import APIRouter, HTTPException, status
from tortoise.exceptions import IntegrityError

from api.models.sensor import SensorDataRequest, SensorDataResponse
from api.database.models import SensorReading

router = APIRouter()


@router.post(
    "/sensors/data",
    response_model=SensorDataResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Store sensor data",
    description="Store sensor reading from IoT device into the database"
)
async def store_sensor_data(sensor_data: SensorDataRequest):
    """
    Store sensor data from IoT device.

    Args:
        sensor_data: Sensor reading data including device_id, timestamp, humidity, temperature

    Returns:
        SensorDataResponse: Stored sensor data with database ID

    Raises:
        400: Invalid sensor data
        409: Duplicate timestamp for device_id
        500: Database error
    """
    try:
        # Create sensor reading in database
        sensor_reading = await SensorReading.create(
            device_id=sensor_data.device_id,
            timestamp=sensor_data.timestamp,
            humidity_raw=sensor_data.humidity_raw,
            humidity_percent=sensor_data.humidity_percent,
            temperature=sensor_data.temperature
        )

        # Return response
        return SensorDataResponse(
            id=sensor_reading.id,
            device_id=sensor_reading.device_id,
            timestamp=sensor_reading.timestamp,
            humidity_percent=float(sensor_reading.humidity_percent),
            temperature=float(sensor_reading.temperature),
            message="Sensor data stored successfully"
        )

    except IntegrityError as e:
        # Handle duplicate timestamp
        if "unique_together" in str(e) or "unique constraint" in str(e).lower():
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Sensor reading already exists for device '{sensor_data.device_id}' at timestamp '{sensor_data.timestamp}'"
            )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Data integrity error: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to store sensor data: {str(e)}"
        )


@router.get(
    "/sensors/{device_id}/latest",
    response_model=SensorDataResponse,
    summary="Get latest sensor reading",
    description="Retrieve the most recent sensor reading for a device"
)
async def get_latest_sensor_data(device_id: str):
    """
    Get the latest sensor reading for a device.

    Args:
        device_id: Sensor device identifier

    Returns:
        SensorDataResponse: Latest sensor reading

    Raises:
        404: No sensor data found for device
    """
    sensor_reading = await SensorReading.filter(device_id=device_id).order_by('-timestamp').first()

    if not sensor_reading:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No sensor data found for device '{device_id}'"
        )

    return SensorDataResponse(
        id=sensor_reading.id,
        device_id=sensor_reading.device_id,
        timestamp=sensor_reading.timestamp,
        humidity_percent=float(sensor_reading.humidity_percent),
        temperature=float(sensor_reading.temperature),
        message="Latest sensor data retrieved successfully"
    )
