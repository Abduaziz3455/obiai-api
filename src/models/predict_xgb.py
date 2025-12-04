"""
Make Predictions with Trained XGBoost Models
"""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path


def load_models():
    """Load trained models and configuration"""
    print("📂 Loading models...")
    
    # Load models
    classifier = joblib.load('models/xgb_classifier.pkl')
    regressor_water = joblib.load('models/xgb_regressor_water.pkl')
    regressor_time = joblib.load('models/xgb_regressor_time.pkl')
    
    # Load config
    with open('models/model_config.json', 'r') as f:
        config = json.load(f)
    
    print(f"✓ Loaded models")
    print(f"  Threshold: {config['threshold']:.2f}")
    print(f"  Features: {len(config['feature_cols'])}")
    
    return classifier, regressor_water, regressor_time, config


def predict(X, classifier, regressor_water, regressor_time, threshold):
    """
    Make irrigation predictions
    
    Returns:
        predictions: DataFrame with columns:
            - irrigation_needed (0/1)
            - recommended_water_percent (0-100)
            - irrigation_time_min (0-300)
            - confidence (0-1)
    """
    # Stage 1: Classify (irrigate yes/no)
    y_pred_proba = classifier.predict_proba(X)[:, 1]
    y_pred_binary = (y_pred_proba >= threshold).astype(int)
    
    # Stage 2: Predict amounts (only for irrigation cases)
    y_pred_water = np.zeros(len(X))
    y_pred_time = np.zeros(len(X))
    
    irrigation_mask = y_pred_binary == 1
    
    if irrigation_mask.sum() > 0:
        y_pred_water[irrigation_mask] = regressor_water.predict(X[irrigation_mask])
        y_pred_time[irrigation_mask] = regressor_time.predict(X[irrigation_mask])
        
        # Clip to valid ranges
        y_pred_water = np.clip(y_pred_water, 0, 100)
        y_pred_time = np.clip(y_pred_time, 0, 300)
    
    # Create results DataFrame
    predictions = pd.DataFrame({
        'irrigation_needed': y_pred_binary,
        'recommended_water_percent': y_pred_water,
        'irrigation_time_min': y_pred_time,
        'confidence': y_pred_proba
    })
    
    return predictions


def predict_from_file(input_file, output_file=None):
    """
    Make predictions from a CSV file
    
    Args:
        input_file: Path to CSV with features
        output_file: Path to save predictions (optional)
    """
    print(f"\n{'='*80}")
    print("🔮 MAKING PREDICTIONS")
    print("="*80)
    
    # Load models
    classifier, regressor_water, regressor_time, config = load_models()
    
    # Load data
    print(f"\n📂 Loading data from: {input_file}")
    df = pd.read_csv(input_file)
    
    # Check if timestamp column exists
    has_timestamp = 'timestamp' in df.columns
    if has_timestamp:
        timestamps = df['timestamp']
        df = df.drop('timestamp', axis=1)
    
    # Ensure correct features
    feature_cols = config['feature_cols']
    missing_features = set(feature_cols) - set(df.columns)
    
    if missing_features:
        print(f"⚠️  Missing features: {missing_features}")
        print("Adding missing features with value 0")
        for feat in missing_features:
            df[feat] = 0
    
    # Select and order features correctly
    X = df[feature_cols]
    
    print(f"✓ Loaded {len(X):,} samples with {len(feature_cols)} features")
    
    # Make predictions
    print(f"\n🔮 Predicting...")
    predictions = predict(X, classifier, regressor_water, regressor_time, config['threshold'])
    
    # Add timestamp back if it existed
    if has_timestamp:
        predictions.insert(0, 'timestamp', timestamps)
    
    # Print summary
    n_irrigation = (predictions['irrigation_needed'] == 1).sum()
    print(f"\n📊 PREDICTION SUMMARY:")
    print(f"  Total samples: {len(predictions):,}")
    print(f"  Irrigation recommended: {n_irrigation:,} ({n_irrigation/len(predictions)*100:.2f}%)")
    
    if n_irrigation > 0:
        irrigation_cases = predictions[predictions['irrigation_needed'] == 1]
        print(f"\n  Irrigation amounts:")
        print(f"    Water %: {irrigation_cases['recommended_water_percent'].mean():.1f}% (avg)")
        print(f"    Duration: {irrigation_cases['irrigation_time_min'].mean():.0f} min (avg)")
        print(f"    Confidence: {irrigation_cases['confidence'].mean():.2f} (avg)")
    
    # Save if requested
    if output_file:
        predictions.to_csv(output_file, index=False)
        print(f"\n💾 Saved predictions to: {output_file}")
    
    return predictions


def predict_single_sample(features_dict):
    """
    Make prediction for a single sample
    
    Args:
        features_dict: Dictionary of feature values
        
    Example:
        features = {
            'temperature_2m': 32.5,
            'precipitation': 0.0,
            'soil_moisture': 0.20,
            'crop_stage': 2,
            'days_since_rain': 8,
            ...
        }
        result = predict_single_sample(features)
    """
    # Load models
    classifier, regressor_water, regressor_time, config = load_models()
    
    # Create DataFrame
    X = pd.DataFrame([features_dict])
    
    # Ensure all features present
    for feat in config['feature_cols']:
        if feat not in X.columns:
            X[feat] = 0
    
    # Select and order correctly
    X = X[config['feature_cols']]
    
    # Predict
    predictions = predict(X, classifier, regressor_water, regressor_time, config['threshold'])
    
    return predictions.iloc[0].to_dict()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python predict.py <input_file.csv> [output_file.csv]")
        print("\nExample:")
        print("  python predict.py data/processed/karmana_ml_ready.csv predictions.csv")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else 'predictions.csv'
    
    predictions = predict_from_file(input_file, output_file)
    
    print("\n✅ Done!")