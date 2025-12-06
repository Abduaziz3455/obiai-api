"""
Prediction service for irrigation recommendations.
"""
import pandas as pd
import numpy as np
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
            Dict with irrigation_needed, recommended_water_percent, irrigation_time_min,
            confidence_score

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

            irrigation_needed = int(predictions['irrigation_needed'].iloc[0])
            irrigation_probability = float(predictions['confidence'].iloc[0])

            # Calculate actual confidence score:
            # - If irrigation_needed = 1: confidence = irrigation_probability
            # - If irrigation_needed = 0: confidence = 1 - irrigation_probability
            # This represents how confident the model is in its prediction
            if irrigation_needed == 1:
                confidence_score = irrigation_probability
            else:
                confidence_score = 1.0 - irrigation_probability

            # Convert to dict with proper types
            recommended_water = float(predictions['recommended_water_percent'].iloc[0])
            irrigation_time = float(predictions['irrigation_time_min'].iloc[0])

            # Sanitize values - replace NaN/inf with 0
            if np.isnan(recommended_water) or np.isinf(recommended_water):
                recommended_water = 0.0
            if np.isnan(irrigation_time) or np.isinf(irrigation_time):
                irrigation_time = 0.0
            if np.isnan(irrigation_probability) or np.isinf(irrigation_probability):
                irrigation_probability = 0.5
                confidence_score = 0.5

            result = {
                'irrigation_needed': irrigation_needed,
                'recommended_water_percent': recommended_water,
                'irrigation_time_min': irrigation_time,
                'confidence_score': confidence_score
            }

            return result

        except Exception as e:
            raise ValueError(f"Prediction failed: {str(e)}")
