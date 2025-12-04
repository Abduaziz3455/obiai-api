"""
Complete XGBoost Model Training for Irrigation Prediction
Two-stage approach: Classification + Regression
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    r2_score, mean_absolute_error, mean_squared_error,
    precision_recall_curve
)
import warnings
warnings.filterwarnings('ignore')

# Check if XGBoost is installed
try:
    import xgboost as xgb
    print("✓ XGBoost installed")
except ImportError:
    print("❌ XGBoost not installed!")
    print("Install with: pip install xgboost")
    exit(1)


def load_and_prepare_data(filepath='data/processed/karmana_ml_ready.csv'):
    """Load and prepare data for training"""
    print("\n" + "="*80)
    print("📂 LOADING DATA")
    print("="*80)
    
    df = pd.read_csv(filepath, parse_dates=['timestamp'])
    print(f"✓ Loaded {len(df):,} samples")
    print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    # Separate features and labels
    feature_cols = [col for col in df.columns 
                   if col not in ['timestamp', 'recommended_water_percent', 'irrigation_time_min']]
    
    X = df[feature_cols].copy()
    y = df[['recommended_water_percent', 'irrigation_time_min']].copy()
    
    print(f"✓ Features: {len(feature_cols)}")
    print(f"✓ Labels: {y.columns.tolist()}")
    
    # Check for missing values
    if X.isnull().sum().sum() > 0:
        print("⚠️  Found missing values, filling with 0...")
        X = X.fillna(0)
    
    return X, y, feature_cols


def create_train_test_split(X, y):
    """Create stratified train/test split"""
    print("\n" + "="*80)
    print("✂️  SPLITTING DATA")
    print("="*80)
    
    # Create binary labels for stratification
    y_binary = (y['irrigation_time_min'] > 0).astype(int)
    
    # Split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y_binary,
        random_state=42
    )
    
    # Also split binary labels
    y_train_binary = (y_train['irrigation_time_min'] > 0).astype(int)
    y_test_binary = (y_test['irrigation_time_min'] > 0).astype(int)
    
    print(f"Training set: {len(X_train):,} samples")
    print(f"  - No irrigation: {(y_train_binary == 0).sum():,}")
    print(f"  - Irrigation: {(y_train_binary == 1).sum():,}")
    print(f"  - Irrigation %: {y_train_binary.sum() / len(y_train_binary) * 100:.2f}%")
    
    print(f"\nTest set: {len(X_test):,} samples")
    print(f"  - No irrigation: {(y_test_binary == 0).sum():,}")
    print(f"  - Irrigation: {(y_test_binary == 1).sum():,}")
    print(f"  - Irrigation %: {y_test_binary.sum() / len(y_test_binary) * 100:.2f}%")
    
    return X_train, X_test, y_train, y_test, y_train_binary, y_test_binary


def train_stage1_classifier(X_train, y_train_binary, X_test, y_test_binary):
    """Stage 1: Train XGBoost classifier for irrigation yes/no"""
    print("\n" + "="*80)
    print("🎯 STAGE 1: IRRIGATION DECISION (YES/NO)")
    print("="*80)
    
    # Calculate scale_pos_weight
    n_negative = (y_train_binary == 0).sum()
    n_positive = (y_train_binary == 1).sum()
    scale_pos_weight = n_negative / n_positive
    
    print(f"\nClass imbalance:")
    print(f"  Negative (no irrigation): {n_negative:,}")
    print(f"  Positive (irrigation): {n_positive:,}")
    print(f"  Ratio: {scale_pos_weight:.1f}:1")
    print(f"  Using scale_pos_weight: {scale_pos_weight:.1f}")
    
    # Train classifier
    print(f"\n🔧 Training XGBoost Classifier...")
    
    classifier = xgb.XGBClassifier(
        n_estimators=300,           # More trees for better learning
        max_depth=8,                # Deep enough to capture patterns
        learning_rate=0.05,         # Lower learning rate with more trees
        scale_pos_weight=scale_pos_weight,  # Handle imbalance
        subsample=0.8,              # Use 80% of data per tree
        colsample_bytree=0.8,       # Use 80% of features per tree
        gamma=1,                    # Minimum loss reduction for split
        min_child_weight=3,         # Minimum samples per leaf
        reg_alpha=0.1,              # L1 regularization
        reg_lambda=1.0,             # L2 regularization
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss'
    )
    
    # Train with early stopping
    eval_set = [(X_train, y_train_binary), (X_test, y_test_binary)]
    
    classifier.fit(
        X_train, y_train_binary,
        eval_set=eval_set,
        verbose=False
    )
    
    print(f"✓ Training complete!")
    print(f"  Trees trained: {classifier.n_estimators}")
    
    return classifier


def optimize_threshold(classifier, X_test, y_test_binary):
    """Find optimal classification threshold"""
    print("\n" + "="*80)
    print("🎚️  OPTIMIZING CLASSIFICATION THRESHOLD")
    print("="*80)
    
    # Get prediction probabilities
    y_pred_proba = classifier.predict_proba(X_test)[:, 1]
    
    # Test different thresholds
    thresholds = np.arange(0.05, 0.6, 0.05)
    
    results = []
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        # Calculate metrics
        from sklearn.metrics import recall_score, precision_score, f1_score
        
        recall = recall_score(y_test_binary, y_pred)
        precision = precision_score(y_test_binary, y_pred, zero_division=0)
        f1 = f1_score(y_test_binary, y_pred, zero_division=0)
        
        results.append({
            'threshold': threshold,
            'recall': recall,
            'precision': precision,
            'f1': f1
        })
    
    results_df = pd.DataFrame(results)
    
    # Print results
    print("\nThreshold optimization results:")
    print(results_df.to_string(index=False))
    
    # Find best threshold (prioritize recall, but keep some precision)
    # Use F0.5 score (weights recall 2x more than precision)
    results_df['f0.5'] = (1 + 0.5**2) * (results_df['precision'] * results_df['recall']) / \
                         (0.5**2 * results_df['precision'] + results_df['recall'])
    
    best_idx = results_df['f0.5'].idxmax()
    best_threshold = results_df.loc[best_idx, 'threshold']
    
    print(f"\n{'='*80}")
    print(f"🎯 OPTIMAL THRESHOLD: {best_threshold:.2f}")
    print(f"   Recall: {results_df.loc[best_idx, 'recall']:.3f}")
    print(f"   Precision: {results_df.loc[best_idx, 'precision']:.3f}")
    print(f"   F1: {results_df.loc[best_idx, 'f1']:.3f}")
    print("="*80)
    
    # Plot threshold analysis
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(results_df['threshold'], results_df['recall'], 'b-', label='Recall', linewidth=2)
    plt.plot(results_df['threshold'], results_df['precision'], 'r-', label='Precision', linewidth=2)
    plt.plot(results_df['threshold'], results_df['f1'], 'g-', label='F1', linewidth=2)
    plt.axvline(best_threshold, color='black', linestyle='--', label='Optimal')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Threshold vs Metrics')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    precision, recall, _ = precision_recall_curve(y_test_binary, y_pred_proba)
    plt.plot(recall, precision, linewidth=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/threshold_optimization.png', dpi=150, bbox_inches='tight')
    print("✓ Saved plot: results/threshold_optimization.png")
    
    return best_threshold


def evaluate_classifier(classifier, X_test, y_test_binary, threshold):
    """Evaluate classifier with optimal threshold"""
    print("\n" + "="*80)
    print("📊 STAGE 1 EVALUATION")
    print("="*80)
    
    # Predict with optimal threshold
    y_pred_proba = classifier.predict_proba(X_test)[:, 1]
    y_pred_binary = (y_pred_proba >= threshold).astype(int)
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test_binary, y_pred_binary,
                               target_names=['No Irrigation', 'Irrigation'],
                               digits=3))
    
    # Confusion matrix
    cm = confusion_matrix(y_test_binary, y_pred_binary)
    
    print("\nConfusion Matrix:")
    print(f"                 Predicted No    Predicted Yes")
    print(f"Actual No        {cm[0,0]:>12,}    {cm[0,1]:>13,}")
    print(f"Actual Yes       {cm[1,0]:>12,}    {cm[1,1]:>13,}")
    
    print(f"\nInterpretation:")
    print(f"  ✓ True Negatives: {cm[0,0]:,} (correctly predicted no irrigation)")
    print(f"  ✗ False Positives: {cm[0,1]:,} (false alarm - wasted water)")
    print(f"  ✗ False Negatives: {cm[1,0]:,} (CRITICAL - missed irrigation!)")
    print(f"  ✓ True Positives: {cm[1,1]:,} (correctly caught irrigation)")
    
    # Calculate key metrics
    recall = cm[1,1] / (cm[1,0] + cm[1,1]) if (cm[1,0] + cm[1,1]) > 0 else 0
    precision = cm[1,1] / (cm[0,1] + cm[1,1]) if (cm[0,1] + cm[1,1]) > 0 else 0
    
    print(f"\n🎯 KEY METRICS:")
    print(f"  Recall: {recall:.1%} - Caught {recall:.1%} of irrigation needs")
    print(f"  Precision: {precision:.1%} - {precision:.1%} of predictions were correct")
    
    return y_pred_binary, y_pred_proba


def train_stage2_regressors(X_train, y_train):
    """Stage 2: Train regressors for irrigation amounts"""
    print("\n" + "="*80)
    print("💧 STAGE 2: IRRIGATION AMOUNT (WHEN NEEDED)")
    print("="*80)
    
    # Filter to irrigation events only
    irrigation_mask = y_train['irrigation_time_min'] > 0
    X_train_irrig = X_train[irrigation_mask]
    y_train_irrig = y_train[irrigation_mask]
    
    print(f"Training on {len(X_train_irrig):,} irrigation events")
    
    # Train water % regressor
    print("\n🔧 Training Water % Regressor...")
    regressor_water = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0,
        min_child_weight=1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1
    )
    
    regressor_water.fit(X_train_irrig, y_train_irrig['recommended_water_percent'])
    print("✓ Water % regressor trained")
    
    # Train time regressor
    print("\n🔧 Training Duration Regressor...")
    regressor_time = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0,
        min_child_weight=1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1
    )
    
    regressor_time.fit(X_train_irrig, y_train_irrig['irrigation_time_min'])
    print("✓ Duration regressor trained")
    
    return regressor_water, regressor_time


def evaluate_regressors(regressor_water, regressor_time, X_test, y_test, y_pred_binary):
    """Evaluate regression models on irrigation events"""
    print("\n" + "="*80)
    print("📊 STAGE 2 EVALUATION")
    print("="*80)
    
    # Predict amounts for all test samples
    y_pred_water_full = np.zeros(len(X_test))
    y_pred_time_full = np.zeros(len(X_test))
    
    # Only predict for samples classified as "irrigation needed"
    irrigation_predicted = y_pred_binary == 1
    
    if irrigation_predicted.sum() > 0:
        y_pred_water_full[irrigation_predicted] = regressor_water.predict(X_test[irrigation_predicted])
        y_pred_time_full[irrigation_predicted] = regressor_time.predict(X_test[irrigation_predicted])
        
        # Clip predictions to valid ranges
        y_pred_water_full = np.clip(y_pred_water_full, 0, 100)
        y_pred_time_full = np.clip(y_pred_time_full, 0, 300)
    
    # Evaluate on actual irrigation events only
    irrigation_actual = y_test['irrigation_time_min'] > 0
    
    if irrigation_actual.sum() > 0:
        y_test_irrig = y_test[irrigation_actual]
        y_pred_water_irrig = y_pred_water_full[irrigation_actual]
        y_pred_time_irrig = y_pred_time_full[irrigation_actual]
        
        # Calculate metrics for water %
        r2_water = r2_score(y_test_irrig['recommended_water_percent'], y_pred_water_irrig)
        mae_water = mean_absolute_error(y_test_irrig['recommended_water_percent'], y_pred_water_irrig)
        rmse_water = np.sqrt(mean_squared_error(y_test_irrig['recommended_water_percent'], y_pred_water_irrig))
        
        # Calculate metrics for time
        r2_time = r2_score(y_test_irrig['irrigation_time_min'], y_pred_time_irrig)
        mae_time = mean_absolute_error(y_test_irrig['irrigation_time_min'], y_pred_time_irrig)
        rmse_time = np.sqrt(mean_squared_error(y_test_irrig['irrigation_time_min'], y_pred_time_irrig))
        
        print(f"\n💧 WATER % PREDICTIONS:")
        print(f"  R² Score: {r2_water:.3f}")
        print(f"  MAE: {mae_water:.2f}%")
        print(f"  RMSE: {rmse_water:.2f}%")
        print(f"  Actual range: {y_test_irrig['recommended_water_percent'].min():.1f}% - {y_test_irrig['recommended_water_percent'].max():.1f}%")
        print(f"  Predicted range: {y_pred_water_irrig.min():.1f}% - {y_pred_water_irrig.max():.1f}%")
        
        print(f"\n⏱️  DURATION PREDICTIONS:")
        print(f"  R² Score: {r2_time:.3f}")
        print(f"  MAE: {mae_time:.1f} minutes")
        print(f"  RMSE: {rmse_time:.1f} minutes")
        print(f"  Actual range: {y_test_irrig['irrigation_time_min'].min():.0f} - {y_test_irrig['irrigation_time_min'].max():.0f} min")
        print(f"  Predicted range: {y_pred_time_irrig.min():.0f} - {y_pred_time_irrig.max():.0f} min")
        
        # Plot predictions vs actual
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Water %
        axes[0].scatter(y_test_irrig['recommended_water_percent'], y_pred_water_irrig, alpha=0.6)
        axes[0].plot([0, 100], [0, 100], 'r--', linewidth=2, label='Perfect prediction')
        axes[0].set_xlabel('Actual Water %')
        axes[0].set_ylabel('Predicted Water %')
        axes[0].set_title(f'Water % Predictions (R²={r2_water:.3f})')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Duration
        axes[1].scatter(y_test_irrig['irrigation_time_min'], y_pred_time_irrig, alpha=0.6)
        max_time = y_test_irrig['irrigation_time_min'].max()
        axes[1].plot([0, max_time], [0, max_time], 'r--', linewidth=2, label='Perfect prediction')
        axes[1].set_xlabel('Actual Duration (min)')
        axes[1].set_ylabel('Predicted Duration (min)')
        axes[1].set_title(f'Duration Predictions (R²={r2_time:.3f})')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/regression_predictions.png', dpi=150, bbox_inches='tight')
        print("\n✓ Saved plot: results/regression_predictions.png")
        
        return r2_water, r2_time
    else:
        print("⚠️  No actual irrigation events in test set!")
        return None, None


def plot_feature_importance(classifier, regressor_water, feature_cols):
    """Plot feature importance from models"""
    print("\n" + "="*80)
    print("📈 FEATURE IMPORTANCE ANALYSIS")
    print("="*80)
    
    # Get feature importances
    importance_class = pd.DataFrame({
        'feature': feature_cols,
        'importance': classifier.feature_importances_
    }).sort_values('importance', ascending=False)
    
    importance_reg = pd.DataFrame({
        'feature': feature_cols,
        'importance': regressor_water.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 features for CLASSIFICATION:")
    print(importance_class.head(10).to_string(index=False))
    
    print("\nTop 10 features for REGRESSION:")
    print(importance_reg.head(10).to_string(index=False))
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Classification
    top_15_class = importance_class.head(15)
    axes[0].barh(range(len(top_15_class)), top_15_class['importance'])
    axes[0].set_yticks(range(len(top_15_class)))
    axes[0].set_yticklabels(top_15_class['feature'])
    axes[0].set_xlabel('Importance')
    axes[0].set_title('Top 15 Features - Classification (Irrigate Yes/No)')
    axes[0].invert_yaxis()
    
    # Regression
    top_15_reg = importance_reg.head(15)
    axes[1].barh(range(len(top_15_reg)), top_15_reg['importance'])
    axes[1].set_yticks(range(len(top_15_reg)))
    axes[1].set_yticklabels(top_15_reg['feature'])
    axes[1].set_xlabel('Importance')
    axes[1].set_title('Top 15 Features - Regression (Water Amount)')
    axes[1].invert_yaxis()
    
    plt.tight_layout()
    plt.savefig('results/feature_importance.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved plot: results/feature_importance.png")


def save_models(classifier, regressor_water, regressor_time, threshold, feature_cols):
    """Save trained models"""
    print("\n" + "="*80)
    print("💾 SAVING MODELS")
    print("="*80)
    
    import joblib
    from pathlib import Path
    
    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)
    
    # Save models
    joblib.dump(classifier, 'models/xgb_classifier.pkl')
    joblib.dump(regressor_water, 'models/xgb_regressor_water.pkl')
    joblib.dump(regressor_time, 'models/xgb_regressor_time.pkl')
    
    # Save threshold and feature names
    import json
    config = {
        'threshold': threshold,
        'feature_cols': feature_cols,
        'model_type': 'xgboost_two_stage'
    }
    
    with open('models/model_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("✓ Saved models:")
    print("  - models/xgb_classifier.pkl")
    print("  - models/xgb_regressor_water.pkl")
    print("  - models/xgb_regressor_time.pkl")
    print("  - models/model_config.json")


def main():
    """Main training pipeline"""
    print("\n" + "="*80)
    print("🌾 IRRIGATION PREDICTION - XGBOOST TRAINING PIPELINE")
    print("="*80)
    
    # Create results directory
    from pathlib import Path
    Path('results').mkdir(exist_ok=True)
    Path('models').mkdir(exist_ok=True)
    
    # Load data
    X, y, feature_cols = load_and_prepare_data()
    
    # Train/test split
    X_train, X_test, y_train, y_test, y_train_binary, y_test_binary = \
        create_train_test_split(X, y)
    
    # Stage 1: Classification
    classifier = train_stage1_classifier(X_train, y_train_binary, X_test, y_test_binary)
    
    # Optimize threshold
    best_threshold = optimize_threshold(classifier, X_test, y_test_binary)
    
    # Evaluate classification
    y_pred_binary, y_pred_proba = evaluate_classifier(
        classifier, X_test, y_test_binary, best_threshold
    )
    
    # Stage 2: Regression
    regressor_water, regressor_time = train_stage2_regressors(X_train, y_train)
    
    # Evaluate regression
    r2_water, r2_time = evaluate_regressors(
        regressor_water, regressor_time, X_test, y_test, y_pred_binary
    )
    
    # Feature importance
    plot_feature_importance(classifier, regressor_water, feature_cols)
    
    # Save models
    save_models(classifier, regressor_water, regressor_time, best_threshold, feature_cols)
    
    # Final summary
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETE!")
    print("="*80)
    print(f"\n📊 FINAL PERFORMANCE SUMMARY:")
    print(f"  Stage 1 (Classification):")
    print(f"    - Optimal threshold: {best_threshold:.2f}")
    print(f"  Stage 2 (Regression on irrigation events):")
    if r2_water is not None:
        print(f"    - Water % R²: {r2_water:.3f}")
        print(f"    - Duration R²: {r2_time:.3f}")
    
    print(f"\n📁 OUTPUT FILES:")
    print(f"  Models: models/")
    print(f"  Plots: results/")
    
    print("\n🎯 Next steps:")
    print("  1. Review plots in results/ folder")
    print("  2. If R² is negative, try lower threshold (0.10-0.15)")
    print("  3. Check feature_importance.png - are soil_moisture, days_since_rain top features?")
    print("  4. If still poor, consider rule-based hybrid approach")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()