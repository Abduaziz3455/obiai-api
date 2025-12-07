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
    yield
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
