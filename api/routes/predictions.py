"""
Prediction endpoints for irrigation recommendations.
"""
from datetime import datetime, timedelta
from fastapi import APIRouter, HTTPException, status

from api.models.prediction import (
    PredictionRequest,
    PredictionResponse,
    SensorDataSummary,
    WeatherSummary
)
from api.database.models import SensorReading
from api.services.weather_service import WeatherService
from api.services.feature_service import FeatureService
from api.services.prediction_service import PredictionService
from api.dependencies import get_models

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
        time_diff = datetime.utcnow() - sensor_reading.timestamp.replace(tzinfo=None)
        if time_diff > timedelta(hours=24):
            # Log warning but continue
            print(f"Warning: Sensor data is {time_diff.total_seconds() / 3600:.1f} hours old")

        # Prepare sensor data dict
        sensor_data = {
            "device_id": sensor_reading.device_id,
            "timestamp": sensor_reading.timestamp,
            "humidity_percent": float(sensor_reading.humidity_percent),
            "temperature": float(sensor_reading.temperature)
        }

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
                planting_date=planting_date
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
            timestamp=sensor_data["timestamp"],
            soil_moisture=sensor_data["humidity_percent"],
            soil_temperature=sensor_data["temperature"]
        )

        weather_summary = WeatherSummary(
            air_temperature=float(latest_weather['temperature_2m']),
            precipitation_24h=float(features_df['precip_24h_sum'].iloc[0]),
            wind_speed=float(latest_weather['wind_speed_10m']),
            solar_radiation=float(latest_weather['shortwave_radiation'])
        )

        response = PredictionResponse(
            irrigation_needed=prediction['irrigation_needed'],
            recommended_water_percent=prediction['recommended_water_percent'],
            irrigation_time_min=prediction['irrigation_time_min'],
            confidence=prediction['confidence'],
            sensor_data=sensor_summary,
            weather_summary=weather_summary,
            timestamp=datetime.utcnow()
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
