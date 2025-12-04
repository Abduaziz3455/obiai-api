import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# Load data
df = pd.read_csv('data/processed/karmana_ml_ready.csv')

# Prepare features and labels
feature_cols = [col for col in df.columns 
                if col not in ['timestamp', 'recommended_water_percent', 'irrigation_time_min']]
X = df[feature_cols]
y = df[['recommended_water_percent', 'irrigation_time_min']]

# Stratified split (ensures both splits have similar irrigation ratios)
irrigation_binary = (y['irrigation_time_min'] > 0).astype(int)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=irrigation_binary, random_state=42
)

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, r2_score

# STAGE 1: Classification (Irrigate yes/no)
print("="*60)
print("STAGE 1: IRRIGATION DECISION (YES/NO)")
print("="*60)

# Create binary labels
y_train_binary = (y_train['irrigation_time_min'] > 0).astype(int)
y_test_binary = (y_test['irrigation_time_min'] > 0).astype(int)

# Train classifier with class weights
classifier = RandomForestClassifier(
    n_estimators=200,
    max_depth=25,
    class_weight='balanced',  # Automatically balance classes
    random_state=42,
    n_jobs=-1
)
classifier.fit(X_train, y_train_binary)

# Predict
y_pred_binary = classifier.predict(X_test)

# Evaluate
print("\nClassification Report:")
print(classification_report(y_test_binary, y_pred_binary, 
                          target_names=['No Irrigation', 'Irrigation']))

# STAGE 2: Regression (How much water?)
print("\n" + "="*60)
print("STAGE 2: IRRIGATION AMOUNT (WHEN NEEDED)")
print("="*60)

# Train ONLY on irrigation events
irrigation_train_mask = y_train['irrigation_time_min'] > 0
X_train_irrig = X_train[irrigation_train_mask]
y_train_irrig = y_train[irrigation_train_mask]

# Train regressors
regressor_water = RandomForestRegressor(
    n_estimators=200,
    max_depth=25,
    random_state=42,
    n_jobs=-1
)
regressor_water.fit(X_train_irrig, y_train_irrig['recommended_water_percent'])

regressor_time = RandomForestRegressor(
    n_estimators=200,
    max_depth=25,
    random_state=42,
    n_jobs=-1
)
regressor_time.fit(X_train_irrig, y_train_irrig['irrigation_time_min'])

# COMBINED PREDICTION
y_pred_water_full = np.zeros(len(X_test))
y_pred_time_full = np.zeros(len(X_test))

# Only predict amounts for samples classified as "irrigation needed"
irrigation_predicted = y_pred_binary == 1
if irrigation_predicted.sum() > 0:
    y_pred_water_full[irrigation_predicted] = regressor_water.predict(X_test[irrigation_predicted])
    y_pred_time_full[irrigation_predicted] = regressor_time.predict(X_test[irrigation_predicted])

# Evaluate on irrigation events only
irrigation_test_mask = y_test['irrigation_time_min'] > 0
if irrigation_test_mask.sum() > 0:
    y_test_irrig = y_test[irrigation_test_mask]
    y_pred_irrig_water = y_pred_water_full[irrigation_test_mask]
    y_pred_irrig_time = y_pred_time_full[irrigation_test_mask]
    
    r2_water = r2_score(y_test_irrig['recommended_water_percent'], y_pred_irrig_water)
    r2_time = r2_score(y_test_irrig['irrigation_time_min'], y_pred_irrig_time)
    
    print(f"\nRegression Performance (on irrigation events):")
    print(f"  Water % R²: {r2_water:.3f}")
    print(f"  Time R²: {r2_time:.3f}")

print("\n" + "="*60)