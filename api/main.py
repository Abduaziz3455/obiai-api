"""
FastAPI main application for Irrigation Prediction System.
"""
import os
from contextlib import asynccontextmanager

import yaml
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from tortoise.contrib.fastapi import register_tortoise

from api.database import TORTOISE_ORM
from api.routes import sensors, predictions, health, disease
from api.services.weather_scheduler import weather_scheduler


# Load API configuration
def load_api_config():
    config_path = os.path.join(os.path.dirname(__file__), "../config/api.yaml")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)['api']


api_config = load_api_config()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI.
    Handles startup and shutdown events.
    """
    # Startup: Initialize Tortoise ORM
    print("🚀 Starting up API...")
    print("📦 Initializing Tortoise ORM...")
    # Tortoise ORM will be registered automatically

    # Start weather scheduler
    print("🌤️  Starting weather data scheduler...")
    # Load locations from config file
    try:
        locations_config_path = os.path.join(os.path.dirname(__file__), "../config/locations.yaml")
        with open(locations_config_path, 'r') as f:
            locations_config = yaml.safe_load(f)

        # Add all active locations to the scheduler
        for location_id, location_data in locations_config.get('locations', {}).items():
            if location_data.get('active', True):
                latitude = location_data['latitude']
                longitude = location_data['longitude']
                weather_scheduler.add_location(latitude, longitude)
                print(f"  ✓ Added location: {location_data['name']} ({latitude}, {longitude})")
    except Exception as e:
        print(f"  ⚠️  Failed to load locations from config: {e}")
        print(f"  Using default location: Tashkent")
        weather_scheduler.add_location(41.2995, 69.2401)

    await weather_scheduler.start()

    yield

    # Shutdown: Stop weather scheduler
    print("🛑 Stopping weather data scheduler...")
    await weather_scheduler.stop()

    # Shutdown: Close database connections
    print("👋 Shutting down API...")


# Create FastAPI application
app = FastAPI(
    title=api_config['title'],
    version=api_config['version'],
    description=api_config['description'],
    lifespan=lifespan,
)

# Register Tortoise ORM
register_tortoise(
    app,
    config=TORTOISE_ORM,
    generate_schemas=False,  # Aerich handles schema generation
    add_exception_handlers=True,
)

# Configure CORS
if api_config['cors']['enabled']:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=api_config['cors']['origins'],
        allow_credentials=api_config['cors']['allow_credentials'],
        allow_methods=api_config['cors']['allow_methods'],
        allow_headers=api_config['cors']['allow_headers'],
    )

# Mount static files for disease prediction uploads
static_dir = os.path.join(os.path.dirname(__file__), "../static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Include routers
app.include_router(health.router, prefix="/api/v1", tags=["Health"])
app.include_router(sensors.router, prefix="/api/v1", tags=["Sensors"])
app.include_router(predictions.router, prefix="/api/v1", tags=["Predictions"])
app.include_router(disease.router, prefix="/api/v1", tags=["Disease Prediction"])


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": api_config['title'],
        "version": api_config['version'],
        "description": api_config['description'],
        "docs": "/docs",
        "health": "/api/v1/health"
    }
