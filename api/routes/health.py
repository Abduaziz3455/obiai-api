"""
Health check endpoint for monitoring API status.
"""
from datetime import datetime

from fastapi import APIRouter, status
from tortoise import Tortoise
import httpx

router = APIRouter()


@router.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    """
    Health check endpoint to verify API status.

    Returns:
        dict: System health status including database, models, and weather API
    """
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "database": "unknown",
        "models": "unknown",
        "weather_api": "unknown"
    }

    # Check database connection
    try:
        conn = Tortoise.get_connection("default")
        await conn.execute_query("SELECT 1")
        health_status["database"] = "connected"
    except Exception as e:
        health_status["database"] = f"error: {str(e)}"
        health_status["status"] = "degraded"

    # Check if models are loaded (will be checked in dependencies later)
    try:
        import os
        model_dir = os.path.join(os.path.dirname(__file__), "../../models")
        required_models = [
            "xgb_classifier.pkl",
            "xgb_regressor_water.pkl",
            "xgb_regressor_time.pkl"
        ]
        models_exist = all(os.path.exists(os.path.join(model_dir, model)) for model in required_models)
        health_status["models"] = "loaded" if models_exist else "missing"
    except Exception as e:
        health_status["models"] = f"error: {str(e)}"
        health_status["status"] = "degraded"

    # Check weather API availability
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get("https://api.open-meteo.com/v1/forecast?latitude=40.48&longitude=65.355&current=temperature_2m")
            health_status["weather_api"] = "available" if response.status_code == 200 else f"error: {response.status_code}"
    except Exception as e:
        health_status["weather_api"] = f"error: {str(e)}"
        health_status["status"] = "degraded"

    return health_status
