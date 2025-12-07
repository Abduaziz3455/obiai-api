"""
Disease prediction service using TensorFlow.
"""
import os
from typing import List, Optional
import numpy as np
from PIL import Image
import tensorflow as tf

from api.models.disease import DiseasePrediction


# Disease mapping (aligned with PlantVillage dataset)
DISEASE_MAPPING = {
    "Apple___Apple_scab": "Olma — Qotir kasalligi",
    "Apple___Black_rot": "Olma — Qora chirish",
    "Apple___Cedar_apple_rust": "Olma — Sadr zang kasalligi",
    "Apple___healthy": "Olma — Sog‘lom",

    "Blueberry___healthy": "Ko‘k meva — Sog‘lom",

    "Cherry_(including_sour)___Powdery_mildew": "Olcha — Un shudring",
    "Cherry_(including_sour)___healthy": "Olcha — Sog‘lom",

    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "Makkajo‘xori — Kulrang barg dog‘i",
    "Corn_(maize)___Common_rust_": "Makkajo‘xori — Zang kasalligi",
    "Corn_(maize)___Northern_Leaf_Blight": "Makkajo‘xori — Shimoliy barg kuyishi",
    "Corn_(maize)___healthy": "Makkajo‘xori — Sog‘lom",

    "Grape___Black_rot": "Uzum — Qora chirish",
    "Grape___Esca_(Black_Measles)": "Uzum — Eska (Qora qizamiq)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "Uzum — Barg kuyishi",
    "Grape___healthy": "Uzum — Sog‘lom",

    "Orange___Haunglongbing_(Citrus_greening)": "Apelsin — Sitrus yashillanish kasalligi (HLB)",

    "Peach___Bacterial_spot": "Shaftoli — Bakterial dog‘",
    "Peach___healthy": "Shaftoli — Sog‘lom",

    "Pepper,_bell___Bacterial_spot": "Bulgar qalampiri — Bakterial dog‘",
    "Pepper,_bell___healthy": "Bulgar qalampiri — Sog‘lom",

    "Potato___Early_blight": "Kartoshka — Erta kuyish (Alternarioz)",
    "Potato___Late_blight": "Kartoshka — Kech kuyish (Fitoftoroz)",
    "Potato___healthy": "Kartoshka — Sog‘lom",

    "Raspberry___healthy": "Malina — Sog‘lom",

    "Soybean___healthy": "Soya — Sog‘lom",

    "Squash___Powdery_mildew": "Qovoq — Un shudring",

    "Strawberry___Leaf_scorch": "Qulupnay — Barg kuyishi",
    "Strawberry___healthy": "Qulupnay — Sog‘lom",

    "Tomato___Bacterial_spot": "Pomidor — Bakterial dog‘",
    "Tomato___Early_blight": "Pomidor — Erta kuyish",
    "Tomato___Late_blight": "Pomidor — Kech kuyish",
    "Tomato___Leaf_Mold": "Pomidor — Barg mog‘ori",
    "Tomato___Septoria_leaf_spot": "Pomidor — Septoriya dog‘i",
    "Tomato___Spider_mites Two-spotted_spider_mite": "Pomidor — Ikki nuqtali o‘rgimchak kana",
    "Tomato___Target_Spot": "Pomidor — Nishonli dog‘",
    "Tomato___Tomato_mosaic_virus": "Pomidor — Mozaika virusi",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Pomidor — Sariq barg o‘ralish virusi",
    "Tomato___healthy": "Pomidor — Sog‘lom"
}

CLASS_LABELS = list(DISEASE_MAPPING.keys())


class DiseaseService:
    """Service for plant disease prediction."""

    def __init__(self, model: Optional[tf.keras.Model] = None):
        """
        Initialize disease service.

        Args:
            model: Pre-loaded TensorFlow model
        """
        self.model = model

    def preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Preprocess image for model inference.

        Args:
            image_path: Path to image file

        Returns:
            Preprocessed image array

        Raises:
            ValueError: If image cannot be loaded or processed
        """
        try:
            img = Image.open(image_path).convert('RGB')
            img = img.resize((224, 224))  # Model input size
            img_array = np.array(img) / 255.0  # Normalize to [0,1]
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
            return img_array
        except Exception as e:
            raise ValueError(f"Rasmni qayta ishlashda xatolik: {str(e)}")

    def predict_disease(self, image_path: str) -> List[DiseasePrediction]:
        """
        Predict plant disease from image.

        Args:
            image_path: Path to uploaded image

        Returns:
            List of top 3 disease predictions

        Raises:
            ValueError: If model is not loaded or prediction fails
        """
        if self.model is None:
            raise ValueError("Model yuklanmagan")

        try:
            # Preprocess image
            img_array = self.preprocess_image(image_path)

            # Make prediction
            predictions = self.model.predict(img_array, verbose=0)[0]

            # Check for invalid predictions
            if np.any(np.isnan(predictions)):
                raise ValueError("Model noto'g'ri javob qaytardi")

            # Get top 3 predictions
            top3_indices = np.argsort(predictions)[-3:][::-1]
            top3_predictions = [
                DiseasePrediction(
                    className=DISEASE_MAPPING[CLASS_LABELS[idx]],
                    confidence=float(predictions[idx]) * 100
                )
                for idx in top3_indices
            ]

            return top3_predictions

        except ValueError:
            raise
        except Exception as e:
            raise ValueError(f"Tashxis qilishda xatolik: {str(e)}")
