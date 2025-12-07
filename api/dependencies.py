"""
Dependency injection for FastAPI.
Handles model loading and caching.
"""
import os
import json
from functools import lru_cache
from typing import Optional
import joblib
import tensorflow as tf


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


@lru_cache()
def get_disease_model() -> Optional[tf.keras.Model]:
    """
    Load TensorFlow disease prediction model once at startup and cache in memory.

    Returns:
        tf.keras.Model: Loaded TensorFlow model, or None if model file not found

    Note:
        The model file should be located at: models/best_model.h5
    """
    base_dir = os.path.dirname(os.path.dirname(__file__))
    model_path = os.path.join(base_dir, "models", "best_model.h5")

    if not os.path.exists(model_path):
        print("⚠️  Disease model not found. Disease prediction will be unavailable.")
        return None

    try:
        model = tf.keras.models.load_model(model_path)
        print(f"✅ Disease model loaded successfully from: {model_path}")
        return model
    except Exception as e:
        print(f"❌ Error loading disease model: {e}")
        return None
