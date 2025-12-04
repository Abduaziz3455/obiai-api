"""
Hybrid Prediction: XGBoost Classification + Rule-Based Amounts
"""

import pandas as pd
import numpy as np
import joblib
import json


def calculate_irrigation_amount(row):
    """
    Rule-based irrigation amount - FIXED VERSION
    Uses multiple indicators, not just soil moisture
    """
    crop_stage = int(row.get('crop_stage', 0))
    days_since_rain = row.get('days_since_rain', 0)
    temp_7d = row.get('temp_7d_mean', 25)
    etc_mm_day = row.get('etc_mm_day', 5)
    precip_7d = row.get('precip_7d_sum', 0)
    
    # BASE AMOUNT by crop stage (realistic defaults)
    stage_base_amounts = {
        0: 35,  # Non-growing season (minimal)
        1: 40,  # Germination-Flowering (moderate)
        2: 55,  # Flowering-Boll (critical - most water)
        3: 45   # Maturation (reduce water)
    }
    
    base_amount = stage_base_amounts.get(crop_stage, 40)
    
    # ADJUSTMENT 1: Days since rain (more days = more water)
    if days_since_rain > 14:
        rain_multiplier = 1.3  # Very dry
    elif days_since_rain > 10:
        rain_multiplier = 1.2  # Dry
    elif days_since_rain > 7:
        rain_multiplier = 1.1  # Somewhat dry
    else:
        rain_multiplier = 1.0  # Recent rain
    
    # ADJUSTMENT 2: Temperature (hotter = more water)
    if temp_7d > 35:
        temp_multiplier = 1.25  # Very hot
    elif temp_7d > 30:
        temp_multiplier = 1.15  # Hot
    elif temp_7d > 25:
        temp_multiplier = 1.05  # Warm
    else:
        temp_multiplier = 0.9   # Cool (less water)
    
    # ADJUSTMENT 3: Evapotranspiration (high ET = more water)
    if etc_mm_day > 8:
        et_multiplier = 1.2   # High water loss
    elif etc_mm_day > 6:
        et_multiplier = 1.1   # Medium water loss
    else:
        et_multiplier = 1.0   # Low water loss
    
    # ADJUSTMENT 4: Recent precipitation (less rain = more irrigation)
    if precip_7d < 5:
        precip_multiplier = 1.2   # No rain
    elif precip_7d < 15:
        precip_multiplier = 1.0   # Some rain
    else:
        precip_multiplier = 0.8   # Good rain (reduce irrigation)
    
    # CALCULATE FINAL AMOUNT
    water_percent = base_amount * rain_multiplier * temp_multiplier * et_multiplier * precip_multiplier
    
    # Clip to realistic range
    water_percent = np.clip(water_percent, 35, 85)
    
    # Calculate duration (2.5 min per % for furrow irrigation)
    irrigation_time = water_percent * 2.5
    irrigation_time = np.clip(irrigation_time, 90, 220)
    
    return round(water_percent, 1), round(irrigation_time, 0)

def predict_hybrid(input_file, output_file='predictions_hybrid.csv'):
    """Make predictions using hybrid approach"""
    print("🔮 HYBRID PREDICTION: XGBoost + Rules")
    print("="*60)
    
    # Load classifier
    print("\n📂 Loading models...")
    classifier = joblib.load('xgb_classifier.pkl')
    with open('model_config.json', 'r') as f:
        config = json.load(f)
    
    # Load data
    print(f"📂 Loading data: {input_file}")
    df = pd.read_csv(input_file)
    
    # Get features
    feature_cols = config['feature_cols']
    X = df[feature_cols]
    
    print(f"✓ Loaded {len(X):,} samples")
    
    # Stage 1: Classification (WHEN to irrigate)
    print(f"\n🎯 Stage 1: Predicting irrigation timing...")
    y_pred_proba = classifier.predict_proba(X)[:, 1]
    y_pred_binary = (y_pred_proba >= config['threshold']).astype(int)
    
    n_irrigation = y_pred_binary.sum()
    print(f"  ✓ Irrigation recommended: {n_irrigation:,} times ({n_irrigation/len(X)*100:.2f}%)")
    
    # Stage 2: Rule-based amounts (HOW MUCH)
    print(f"\n💧 Stage 2: Calculating amounts (rule-based)...")
    
    water_amounts = []
    durations = []
    
    for idx, row in df.iterrows():
        if y_pred_binary[idx] == 1:
            water, duration = calculate_irrigation_amount(row)
        else:
            water, duration = 0, 0
        
        water_amounts.append(water)
        durations.append(duration)
    
    # Create results
    predictions = pd.DataFrame({
        'timestamp': df['timestamp'] if 'timestamp' in df.columns else range(len(df)),
        'irrigation_needed': y_pred_binary,
        'recommended_water_percent': water_amounts,
        'irrigation_time_min': durations,
        'confidence': y_pred_proba
    })
    
    # Save
    predictions.to_csv(output_file, index=False)
    print(f"\n✓ Saved predictions: {output_file}")
    
    # Summary
    if n_irrigation > 0:
        irrig_cases = predictions[predictions['irrigation_needed'] == 1]
        print(f"\n📊 SUMMARY:")
        print(f"  Irrigation events: {n_irrigation:,}")
        print(f"  Avg water: {irrig_cases['recommended_water_percent'].mean():.1f}%")
        print(f"  Avg duration: {irrig_cases['irrigation_time_min'].mean():.0f} min")
        print(f"  Avg confidence: {irrig_cases['confidence'].mean():.2f}")
    
    return predictions


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python predict_hybrid.py <input_file.csv> [output_file.csv]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else 'predictions_hybrid.csv'
    
    predictions = predict_hybrid(input_file, output_file)
    print("\n✅ Done!")
