"""
Pydantic models for disease prediction API.
"""
from typing import List, Optional
from pydantic import BaseModel, Field


class DiseasePrediction(BaseModel):
    """Single disease prediction result."""
    className: str = Field(..., description="Disease class name (localized)")
    confidence: float = Field(..., description="Confidence percentage (0-100)", ge=0, le=100)


class DiseasePredictionRequest(BaseModel):
    """Request model for disease prediction."""
    crop: Optional[str] = Field(None, description="Crop type (optional metadata)")

    class Config:
        json_schema_extra = {
            "example": {
                "crop": "Tomato"
            }
        }


class DiseasePredictionResponse(BaseModel):
    """Response model for disease prediction."""
    success: bool = Field(..., description="Whether prediction was successful")
    predictions: List[DiseasePrediction] = Field(..., description="Top 3 disease predictions")
    image_path: Optional[str] = Field(None, description="Saved image path")

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "predictions": [
                    {"className": "Pomidor - Soglom", "confidence": 95.5},
                    {"className": "Pomidor - Erta_kuyish", "confidence": 3.2},
                    {"className": "Pomidor - Kech_kuyish", "confidence": 1.3}
                ],
                "image_path": "uploads/20241207_123456_image.jpg"
            }
        }


class ErrorResponse(BaseModel):
    """Error response model."""
    success: bool = Field(default=False, description="Success status")
    error: str = Field(..., description="Error message")

    class Config:
        json_schema_extra = {
            "example": {
                "success": False,
                "error": "Model mavjud emas"
            }
        }
