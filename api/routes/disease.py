"""
Disease prediction API endpoints.
"""
import os
import base64
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse

from api.dependencies import get_disease_model
from api.services.disease_service import DiseaseService
from api.models.disease import (
    DiseasePredictionResponse,
    ErrorResponse,
)


router = APIRouter()

# Constants
UPLOAD_DIR = "static/uploads"
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB


def is_allowed_file(filename: str) -> bool:
    """Check if file extension is allowed."""
    return os.path.splitext(filename.lower())[1] in ALLOWED_EXTENSIONS


def save_uploaded_file(file: UploadFile) -> str:
    """
    Save uploaded file to disk.

    Args:
        file: Uploaded file object

    Returns:
        Path to saved file

    Raises:
        HTTPException: If file validation fails
    """
    # Validate file extension
    if not is_allowed_file(file.filename):
        raise HTTPException(
            status_code=400,
            detail="Ruxsat etilmagan fayl turi! Faqat JPG, JPEG yoki PNG."
        )

    # Create upload directory if it doesn't exist
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    # Generate unique filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    ext = os.path.splitext(file.filename)[1]
    unique_filename = f"{timestamp}_{file.filename}"
    file_path = os.path.join(UPLOAD_DIR, unique_filename)

    # Save file
    with open(file_path, "wb") as buffer:
        content = file.file.read()
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"Fayl hajmi {MAX_FILE_SIZE / (1024*1024)}MB dan oshmasligi kerak"
            )
        buffer.write(content)

    return file_path


def save_base64_image(image_data: str) -> str:
    """
    Save base64 encoded image to disk.

    Args:
        image_data: Base64 encoded image string

    Returns:
        Path to saved file

    Raises:
        HTTPException: If decoding fails
    """
    try:
        # Remove data URL prefix if present
        if ',' in image_data:
            image_data = image_data.split(',')[1]

        # Decode base64
        image_bytes = base64.b64decode(image_data)

        # Create upload directory if it doesn't exist
        os.makedirs(UPLOAD_DIR, exist_ok=True)

        # Generate unique filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_filename = f"{timestamp}_camera.png"
        file_path = os.path.join(UPLOAD_DIR, unique_filename)

        # Save file
        with open(file_path, 'wb') as f:
            f.write(image_bytes)

        return file_path

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Rasmni yuklashda xatolik: {str(e)}"
        )


@router.post(
    "/diseases/predict",
    response_model=DiseasePredictionResponse,
    summary="Predict plant disease from image",
    description="Upload an image of a plant to detect diseases. Supports file upload or base64 encoded image.",
    responses={
        200: {"description": "Successful prediction"},
        400: {"description": "Invalid input", "model": ErrorResponse},
        500: {"description": "Server error", "model": ErrorResponse},
    }
)
async def predict_disease(
    image: Optional[UploadFile] = File(None, description="Image file (JPG, JPEG, PNG)"),
    camera_image: Optional[str] = Form(None, description="Base64 encoded image from camera"),
    crop: Optional[str] = Form(None, description="Crop type (optional metadata)")
):
    """
    Predict plant disease from uploaded image.

    This endpoint accepts either:
    - A file upload via multipart/form-data
    - A base64 encoded image string via form field

    Returns the top 3 most likely disease predictions with confidence scores.
    """
    try:
        # Load disease model
        model = get_disease_model()
        if model is None:
            raise HTTPException(
                status_code=503,
                detail="Model yuklanmagan. Administrator bilan bog'laning."
            )

        # Initialize service
        disease_service = DiseaseService(model=model)

        # Process image
        image_path = None
        saved_path = None

        if image and image.filename:
            # File upload
            image_path = save_uploaded_file(image)
            saved_path = image_path.replace(UPLOAD_DIR + "/", "uploads/")
        elif camera_image:
            # Base64 image
            image_path = save_base64_image(camera_image)
            saved_path = image_path.replace(UPLOAD_DIR + "/", "uploads/")
        else:
            raise HTTPException(
                status_code=400,
                detail="Rasm tanlanmadi! Fayl yoki camera rasm yuklang."
            )

        # Make prediction
        predictions = disease_service.predict_disease(image_path)

        return DiseasePredictionResponse(
            success=True,
            predictions=predictions,
            image_path=saved_path
        )

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Serverda xatolik: {str(e)}"
        )


@router.get(
    "/diseases/classes",
    summary="Get list of supported disease classes",
    description="Returns all supported plant disease classes with their localized names.",
)
async def get_disease_classes():
    """Get list of all supported disease classes."""
    from api.services.disease_service import DISEASE_MAPPING

    return {
        "success": True,
        "total": len(DISEASE_MAPPING),
        "classes": DISEASE_MAPPING
    }
