"""
Prediction endpoints for irrigation recommendations.
"""
from datetime import datetime, timedelta, timezone
from fastapi import APIRouter, HTTPException, status
import numpy as np
from zoneinfo import ZoneInfo

from api.models.prediction import (
    PredictionRequest,
    PredictionResponse,
    SensorDataSummary,
    WeatherSummary
)
from api.database.models import SensorReading, PredictionHistory
from api.services.weather_service import WeatherService
from api.services.feature_service import FeatureService
from api.services.prediction_service import PredictionService
from api.dependencies import get_models


def safe_float(value, default=0.0, decimals=None):
    """Convert value to float, replacing NaN/inf with default."""
    try:
        val = float(value)
        if np.isnan(val) or np.isinf(val):
            return default
        if decimals is not None:
            return round(val, decimals)
        return val
    except (TypeError, ValueError):
        return default


def to_tashkent_tz(utc_datetime):
    """Convert UTC datetime to Asia/Tashkent timezone."""
    if utc_datetime is None:
        return None
    # Ensure the datetime is timezone-aware (UTC)
    if utc_datetime.tzinfo is None:
        utc_datetime = utc_datetime.replace(tzinfo=timezone.utc)
    # Convert to Tashkent timezone
    return utc_datetime.astimezone(ZoneInfo("Asia/Tashkent"))


router = APIRouter()

# Initialize services
weather_service = WeatherService()
feature_service = FeatureService()


@router.post(
    "/predictions",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict irrigation needs",
    description="Generate irrigation recommendation based on sensor and weather data"
)
async def predict_irrigation(request: PredictionRequest):
    """
    Generate irrigation prediction.

    Process:
    1. Fetch latest sensor data from database
    2. Fetch current and historical weather data from Open-Meteo API
    3. Generate 42 features using feature engineering pipeline
    4. Run two-stage XGBoost prediction
    5. Return irrigation recommendation

    Args:
        request: Prediction request with device_id, location, and crop config

    Returns:
        PredictionResponse: Irrigation recommendation with confidence score

    Raises:
        404: No sensor data found
        503: Weather API unavailable
        500: Prediction error
    """
    try:
        # Step 1: Fetch latest sensor data
        sensor_reading = await SensorReading.filter(
            device_id=request.device_id
        ).order_by('-timestamp').first()

        if not sensor_reading:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No sensor data found for device '{request.device_id}'"
            )

        # Check if sensor data is not too old (warning if > 24 hours)
        from datetime import timezone
        now_utc = datetime.now(timezone.utc)
        sensor_timestamp = sensor_reading.timestamp if sensor_reading.timestamp.tzinfo else sensor_reading.timestamp.replace(tzinfo=timezone.utc)
        time_diff = now_utc - sensor_timestamp
        if time_diff > timedelta(hours=24):
            # Log warning but continue
            print(f"Warning: Sensor data is {time_diff.total_seconds() / 3600:.1f} hours old")

        # Step 1b: Fetch historical sensor data (last 30 days for trends)
        cutoff_time = now_utc - timedelta(days=30)
        historical_readings = await SensorReading.filter(
            device_id=request.device_id,
            timestamp__gte=cutoff_time
        ).order_by('timestamp').all()

        # Prepare sensor data dict with latest reading
        sensor_data = {
            "device_id": sensor_reading.device_id,
            "timestamp": sensor_reading.timestamp,
            "humidity_percent": float(sensor_reading.humidity_percent),
            "temperature": float(sensor_reading.temperature)
        }

        # Add historical sensor readings as a list for trend analysis
        sensor_history = [
            {
                "timestamp": reading.timestamp,
                "humidity_percent": float(reading.humidity_percent),
                "temperature": float(reading.temperature)
            }
            for reading in historical_readings
        ]

        # Step 2: Fetch weather data (current + 30 days historical)
        try:
            weather_df = await weather_service.get_current_and_historical_weather(
                latitude=request.location.latitude,
                longitude=request.location.longitude,
                hours_back=720  # 30 days
            )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Failed to fetch weather data: {str(e)}"
            )

        if weather_df.empty:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Weather API returned no data"
            )

        # Step 3: Generate features
        planting_date = request.crop_config.planting_date if request.crop_config else None

        try:
            features_df = await feature_service.prepare_features(
                sensor_data=sensor_data,
                weather_df=weather_df,
                planting_date=planting_date,
                sensor_history=sensor_history if len(sensor_history) > 1 else None
            )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Feature engineering failed: {str(e)}"
            )

        # Step 4: Load models and make prediction
        try:
            models = get_models()
            prediction_service = PredictionService(models)
            prediction = prediction_service.predict_irrigation(features_df)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Prediction failed: {str(e)}"
            )

        # Step 5: Prepare response
        # Get latest weather data for summary
        latest_weather = weather_df.iloc[-1]

        sensor_summary = SensorDataSummary(
            device_id=sensor_data["device_id"],
            timestamp=to_tashkent_tz(sensor_data["timestamp"]),
            soil_moisture=safe_float(sensor_data["humidity_percent"]),
            soil_temperature=safe_float(sensor_data["temperature"])
        )

        weather_summary = WeatherSummary(
            air_temperature=safe_float(latest_weather['temperature_2m']),
            precipitation_24h=safe_float(features_df['precip_24h_sum'].iloc[0]),
            wind_speed=safe_float(latest_weather['wind_speed_10m']),
            solar_radiation=safe_float(latest_weather['shortwave_radiation'])
        )

        # Convert to Tashkent timezone and remove microseconds for cleaner output
        clean_timestamp = to_tashkent_tz(now_utc.replace(microsecond=0))

        # Save prediction to database
        await PredictionHistory.create(
            device_id=request.device_id,
            sensor_reading_id=sensor_reading.id,
            soil_moisture=safe_float(sensor_data["humidity_percent"]),
            soil_temperature=safe_float(sensor_data["temperature"]),
            air_temperature=safe_float(latest_weather['temperature_2m']),
            irrigation_needed=prediction['irrigation_needed'],
            recommended_water_percent=safe_float(prediction['recommended_water_percent']),
            irrigation_time_min=safe_float(prediction['irrigation_time_min']),
            confidence=safe_float(prediction['confidence_score'], decimals=4),
            features_json=features_df.iloc[0].to_dict()
        )

        response = PredictionResponse(
            irrigation_needed=prediction['irrigation_needed'],
            recommended_water_percent=prediction['recommended_water_percent'],
            irrigation_time_min=prediction['irrigation_time_min'],
            confidence_score=safe_float(prediction['confidence_score'], decimals=5),
            sensor_data=sensor_summary,
            weather_summary=weather_summary,
            timestamp=clean_timestamp
        )

        return response

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Catch-all for unexpected errors
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error: {str(e)}"
        )


@router.get(
    "/predictions/{prediction_id}",
    status_code=status.HTTP_200_OK,
    summary="Get single irrigation prediction by ID",
    description="Retrieve a specific irrigation prediction by its ID"
)
async def get_prediction_by_id(prediction_id: int):
    """
    Get a single irrigation prediction by ID.

    Args:
        prediction_id: The ID of the prediction to retrieve

    Returns:
        Single irrigation prediction record

    Raises:
        404: Prediction not found
        500: Server error
    """
    try:
        prediction = await PredictionHistory.get_or_none(id=prediction_id)

        if not prediction:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Prediction with ID {prediction_id} not found"
            )

        return {
            "success": True,
            "prediction": {
                "id": prediction.id,
                "device_id": prediction.device_id,
                "prediction_timestamp": to_tashkent_tz(prediction.prediction_timestamp),
                "soil_moisture": float(prediction.soil_moisture) if prediction.soil_moisture else None,
                "soil_temperature": float(prediction.soil_temperature) if prediction.soil_temperature else None,
                "air_temperature": float(prediction.air_temperature) if prediction.air_temperature else None,
                "irrigation_needed": prediction.irrigation_needed,
                "recommended_water_percent": float(prediction.recommended_water_percent),
                "irrigation_time_min": float(prediction.irrigation_time_min),
                "confidence": float(prediction.confidence),
                "model_version": prediction.model_version
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch prediction: {str(e)}"
        )


@router.get(
    "/predictions",
    status_code=status.HTTP_200_OK,
    summary="Get irrigation prediction history",
    description="Retrieve historical irrigation predictions with optional filtering and pagination"
)
async def get_predictions(
    device_id: str = None,
    limit: int = 50,
    offset: int = 0,
    irrigation_needed: int = None
):
    """
    Get irrigation prediction history.

    Args:
        device_id: Filter by device ID (optional)
        limit: Maximum number of records to return (default: 50, max: 100)
        offset: Number of records to skip (default: 0)
        irrigation_needed: Filter by irrigation status: 0 (not needed) or 1 (needed)

    Returns:
        List of historical irrigation predictions with pagination info
    """
    try:
        # Enforce maximum limit
        limit = min(limit, 100)

        # Build query
        query = PredictionHistory.all()

        # Apply filters
        if device_id:
            query = query.filter(device_id=device_id)

        if irrigation_needed is not None:
            query = query.filter(irrigation_needed=irrigation_needed)

        # Get total count
        total = await query.count()

        # Get paginated results
        predictions = await query.order_by('-prediction_timestamp').offset(offset).limit(limit)

        # Format response
        results = [
            {
                "id": p.id,
                "device_id": p.device_id,
                "prediction_timestamp": to_tashkent_tz(p.prediction_timestamp),
                "soil_moisture": float(p.soil_moisture) if p.soil_moisture else None,
                "soil_temperature": float(p.soil_temperature) if p.soil_temperature else None,
                "air_temperature": float(p.air_temperature) if p.air_temperature else None,
                "irrigation_needed": p.irrigation_needed,
                "recommended_water_percent": float(p.recommended_water_percent),
                "irrigation_time_min": float(p.irrigation_time_min),
                "confidence": float(p.confidence),
                "model_version": p.model_version
            }
            for p in predictions
        ]

        return {
            "success": True,
            "total": total,
            "limit": limit,
            "offset": offset,
            "count": len(results),
            "predictions": results
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch predictions: {str(e)}"
        )
