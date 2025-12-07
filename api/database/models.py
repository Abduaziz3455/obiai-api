"""
Tortoise ORM models for irrigation predictor database.
"""
from tortoise import fields
from tortoise.models import Model


class SensorReading(Model):
    """Sensor readings from IoT devices."""

    id = fields.BigIntField(pk=True)
    device_id = fields.CharField(max_length=50, index=True)
    timestamp = fields.DatetimeField(index=True)
    humidity_raw = fields.DecimalField(max_digits=5, decimal_places=2)
    humidity_percent = fields.DecimalField(max_digits=5, decimal_places=2)
    temperature = fields.DecimalField(max_digits=5, decimal_places=2)
    created_at = fields.DatetimeField(auto_now_add=True)

    class Meta:
        table = "sensor_readings"
        unique_together = (("device_id", "timestamp"),)
        indexes = [("device_id", "timestamp"), ("timestamp",)]

    def __str__(self):
        return f"SensorReading(device_id={self.device_id}, timestamp={self.timestamp})"


class PredictionHistory(Model):
    """History of irrigation predictions."""

    id = fields.BigIntField(pk=True)
    device_id = fields.CharField(max_length=50, index=True)
    prediction_timestamp = fields.DatetimeField(auto_now_add=True, index=True)
    sensor_reading = fields.ForeignKeyField(
        "models.SensorReading",
        related_name="predictions",
        null=True
    )

    # Model inputs summary
    soil_moisture = fields.DecimalField(max_digits=5, decimal_places=2, null=True)
    soil_temperature = fields.DecimalField(max_digits=5, decimal_places=2, null=True)
    air_temperature = fields.DecimalField(max_digits=5, decimal_places=2, null=True)

    # Predictions
    irrigation_needed = fields.IntField()
    recommended_water_percent = fields.DecimalField(max_digits=5, decimal_places=2)
    irrigation_time_min = fields.DecimalField(max_digits=6, decimal_places=2)
    confidence = fields.DecimalField(max_digits=5, decimal_places=4)

    # Metadata
    model_version = fields.CharField(max_length=20, default="xgboost_v1")
    features_json = fields.JSONField(null=True)

    created_at = fields.DatetimeField(auto_now_add=True)

    class Meta:
        table = "prediction_history"
        indexes = [("device_id", "prediction_timestamp")]

    def __str__(self):
        return f"Prediction(device_id={self.device_id}, irrigation_needed={self.irrigation_needed})"


class DiseasePredictionHistory(Model):
    """History of disease predictions."""

    id = fields.BigIntField(pk=True)
    prediction_timestamp = fields.DatetimeField(auto_now_add=True, index=True)

    # Image info
    image_path = fields.CharField(max_length=255)
    crop_type = fields.CharField(max_length=50, null=True)

    # Top prediction
    predicted_class = fields.CharField(max_length=100)
    confidence = fields.DecimalField(max_digits=5, decimal_places=2)

    # All predictions (JSON)
    all_predictions = fields.JSONField(null=True)

    # Metadata
    model_version = fields.CharField(max_length=20, default="tensorflow_v1")

    created_at = fields.DatetimeField(auto_now_add=True)

    class Meta:
        table = "disease_prediction_history"
        indexes = [("prediction_timestamp",), ("predicted_class",)]

    def __str__(self):
        return f"DiseasePrediction(class={self.predicted_class}, confidence={self.confidence})"


class WeatherCache(Model):
    """Cache for weather API responses."""

    id = fields.BigIntField(pk=True)
    location_key = fields.CharField(max_length=100, index=True)
    timestamp = fields.DatetimeField(index=True)

    # Weather parameters
    temperature_2m = fields.DecimalField(max_digits=5, decimal_places=2, null=True)
    precipitation = fields.DecimalField(max_digits=7, decimal_places=3, null=True)
    wind_speed_10m = fields.DecimalField(max_digits=6, decimal_places=2, null=True)
    shortwave_radiation = fields.DecimalField(max_digits=8, decimal_places=2, null=True)

    # Cache metadata
    fetched_at = fields.DatetimeField(auto_now_add=True)
    expires_at = fields.DatetimeField()

    class Meta:
        table = "weather_cache"
        unique_together = (("location_key", "timestamp"),)
        indexes = [("expires_at",)]

    def __str__(self):
        return f"WeatherCache(location={self.location_key}, timestamp={self.timestamp})"
