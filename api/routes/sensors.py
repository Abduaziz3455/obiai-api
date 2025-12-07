"""
Sensor data endpoints for storing IoT sensor readings.
"""
from fastapi import APIRouter, HTTPException, status
from tortoise.exceptions import IntegrityError

from api.models.sensor import SensorDataRequest, SensorDataResponse, SensorDataListResponse, SensorStatistics
from api.database.models import SensorReading
from tortoise.functions import Min, Max, Avg
from datetime import timezone
from zoneinfo import ZoneInfo

router = APIRouter()


def to_tashkent_tz(utc_datetime):
    """Convert UTC datetime to Asia/Tashkent timezone."""
    if utc_datetime is None:
        return None
    # Ensure the datetime is timezone-aware (UTC)
    if utc_datetime.tzinfo is None:
        utc_datetime = utc_datetime.replace(tzinfo=timezone.utc)
    # Convert to Tashkent timezone
    return utc_datetime.astimezone(ZoneInfo("Asia/Tashkent"))


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
        # Ensure timestamp is timezone-aware (UTC)
        timestamp = sensor_data.timestamp
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)

        # Create sensor reading in database
        sensor_reading = await SensorReading.create(
            device_id=sensor_data.device_id,
            timestamp=timestamp,
            humidity_raw=sensor_data.humidity_raw,
            humidity_percent=sensor_data.humidity_percent,
            temperature=sensor_data.temperature
        )

        # Return response
        return SensorDataResponse(
            id=sensor_reading.id,
            device_id=sensor_reading.device_id,
            timestamp=to_tashkent_tz(sensor_reading.timestamp),
            humidity_raw=float(sensor_reading.humidity_raw),
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
        timestamp=to_tashkent_tz(sensor_reading.timestamp),
        humidity_raw=float(sensor_reading.humidity_raw),
        humidity_percent=float(sensor_reading.humidity_percent),
        temperature=float(sensor_reading.temperature),
        message="Latest sensor data retrieved successfully"
    )


@router.get(
    "/sensors/{device_id}/history",
    response_model=SensorDataListResponse,
    summary="Get sensor reading history",
    description="Retrieve all sensor readings for a device with optional time filters and pagination"
)
async def get_sensor_history(
    device_id: str,
    limit: int = 100,
    offset: int = 0,
    hours_back: int | None = None
):
    """
    Get historical sensor readings for a device.

    Args:
        device_id: Sensor device identifier
        limit: Maximum number of records to return (default: 100, max: 1000)
        offset: Number of records to skip (default: 0)
        hours_back: Optional filter to get data from last N hours

    Returns:
        SensorDataListResponse: List of sensor readings with total count

    Raises:
        400: Invalid parameters
        404: No sensor data found for device
    """
    # Validate parameters
    if limit > 1000:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Limit cannot exceed 1000"
        )

    if limit < 1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Limit must be at least 1"
        )

    if offset < 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Offset cannot be negative"
        )

    # Build query
    query = SensorReading.filter(device_id=device_id)

    # Apply time filter if specified
    if hours_back is not None:
        if hours_back < 1:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="hours_back must be at least 1"
            )
        from datetime import datetime, timedelta, timezone
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours_back)
        query = query.filter(timestamp__gte=cutoff_time)

    # Get total count
    total = await query.count()

    if total == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No sensor data found for device '{device_id}'"
        )

    # Get paginated results, ordered by most recent first
    sensor_readings = await query.order_by('-timestamp').offset(offset).limit(limit)

    # Convert to response format
    data = [
        SensorDataResponse(
            id=reading.id,
            device_id=reading.device_id,
            timestamp=to_tashkent_tz(reading.timestamp),
            humidity_raw=float(reading.humidity_raw),
            humidity_percent=float(reading.humidity_percent),
            temperature=float(reading.temperature),
            message=""
        )
        for reading in sensor_readings
    ]

    return SensorDataListResponse(
        total=total,
        data=data
    )


@router.get(
    "/sensors/{device_id}/statistics",
    response_model=SensorStatistics,
    summary="Get sensor statistics",
    description="Calculate statistics for sensor readings over a time period"
)
async def get_sensor_statistics(
    device_id: str,
    hours_back: int = 24
):
    """
    Get statistical summary of sensor readings.

    Args:
        device_id: Sensor device identifier
        hours_back: Number of hours to analyze (default: 24)

    Returns:
        SensorStatistics: Statistical summary including min, max, avg values

    Raises:
        400: Invalid parameters
        404: No sensor data found for device
    """
    if hours_back < 1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="hours_back must be at least 1"
        )

    from datetime import datetime, timedelta, timezone

    # Calculate time range
    cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours_back)

    # Build query for the time range
    query = SensorReading.filter(
        device_id=device_id,
        timestamp__gte=cutoff_time
    )

    # Get count
    total_readings = await query.count()

    if total_readings == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No sensor data found for device '{device_id}' in the last {hours_back} hours"
        )

    # Get aggregated statistics
    stats = await query.annotate(
        humidity_min=Min('humidity_percent'),
        humidity_max=Max('humidity_percent'),
        humidity_avg=Avg('humidity_percent'),
        temperature_min=Min('temperature'),
        temperature_max=Max('temperature'),
        temperature_avg=Avg('temperature')
    ).values(
        'humidity_min',
        'humidity_max',
        'humidity_avg',
        'temperature_min',
        'temperature_max',
        'temperature_avg'
    )

    # Get the first result (there's only one row with aggregates)
    stats_result = stats[0]

    # Get latest and oldest readings for timestamps and latest values
    latest_reading = await query.order_by('-timestamp').first()
    oldest_reading = await query.order_by('timestamp').first()

    # Calculate actual time range
    time_diff = latest_reading.timestamp - oldest_reading.timestamp
    time_range_hours = time_diff.total_seconds() / 3600

    return SensorStatistics(
        device_id=device_id,
        total_readings=total_readings,
        time_range_hours=round(time_range_hours, 2),
        humidity_min=float(stats_result['humidity_min']),
        humidity_max=float(stats_result['humidity_max']),
        humidity_avg=round(float(stats_result['humidity_avg']), 2),
        humidity_latest=float(latest_reading.humidity_percent),
        temperature_min=float(stats_result['temperature_min']),
        temperature_max=float(stats_result['temperature_max']),
        temperature_avg=round(float(stats_result['temperature_avg']), 2),
        temperature_latest=float(latest_reading.temperature),
        oldest_reading=to_tashkent_tz(oldest_reading.timestamp),
        latest_reading=to_tashkent_tz(latest_reading.timestamp)
    )
