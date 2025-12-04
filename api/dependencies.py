"""
Dependency injection for FastAPI.
Handles model loading and caching.
"""
import os
import json
from functools import lru_cache
import joblib


@lru_cache()
def get_models():
    """
    Load XGBoost models once at startup and cache in memory.

    Returns:
        dict: Dictionary containing classifier, regressors, and config

    Raises:
        FileNotFoundError: If model files are missing
    """
    base_path = os.path.join(os.path.dirname(__file__), "../models")

    model_paths = {
        "classifier": os.path.join(base_path, "xgb_classifier.pkl"),
        "regressor_water": os.path.join(base_path, "xgb_regressor_water.pkl"),
        "regressor_time": os.path.join(base_path, "xgb_regressor_time.pkl"),
        "config": os.path.join(base_path, "model_config.json")
    }

    # Check if all model files exist
    for name, path in model_paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")

    # Load models
    classifier = joblib.load(model_paths["classifier"])
    regressor_water = joblib.load(model_paths["regressor_water"])
    regressor_time = joblib.load(model_paths["regressor_time"])

    # Load config
    with open(model_paths["config"], 'r') as f:
        config = json.load(f)

    return {
        "classifier": classifier,
        "regressor_water": regressor_water,
        "regressor_time": regressor_time,
        "config": config
    }
