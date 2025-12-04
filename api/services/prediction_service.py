"""
Prediction service for irrigation recommendations.
"""
import pandas as pd
from typing import Dict

from src.models.predict_xgb import predict


class PredictionService:
    """
    Service for generating irrigation predictions using XGBoost models.
    """

    def __init__(self, models: Dict):
        """
        Initialize prediction service.

        Args:
            models: Dict containing classifier, regressors, and config
        """
        self.classifier = models['classifier']
        self.regressor_water = models['regressor_water']
        self.regressor_time = models['regressor_time']
        self.threshold = models['config']['threshold']

    def predict_irrigation(self, features_df: pd.DataFrame) -> Dict:
        """
        Generate irrigation prediction using two-stage XGBoost models.

        Args:
            features_df: DataFrame with 42 features in correct order

        Returns:
            Dict with irrigation_needed, recommended_water_percent, irrigation_time_min, confidence

        Raises:
            ValueError: If prediction fails
        """
        try:
            # Use existing predict function from predict_xgb.py
            predictions = predict(
                features_df,
                self.classifier,
                self.regressor_water,
                self.regressor_time,
                self.threshold
            )

            # Convert to dict with proper types
            result = {
                'irrigation_needed': int(predictions['irrigation_needed'].iloc[0]),
                'recommended_water_percent': float(predictions['recommended_water_percent'].iloc[0]),
                'irrigation_time_min': float(predictions['irrigation_time_min'].iloc[0]),
                'confidence': float(predictions['confidence'].iloc[0])
            }

            return result

        except Exception as e:
            raise ValueError(f"Prediction failed: {str(e)}")
