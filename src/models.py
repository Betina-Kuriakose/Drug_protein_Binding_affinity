"""
Machine Learning Models for Drug-Protein Binding Affinity Prediction

This script implements simpler ML models:
- Random Forest Regressor
- XGBoost Regressor
- Linear Regression
- Ridge Regression
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.model_selection import cross_val_score, KFold
import warnings
warnings.filterwarnings('ignore')


def load_features(filepath='dataset/features.csv'):
    """Load feature matrix"""
    print("Loading features...")
    df = pd.read_csv(filepath)
    
    # Separate features and target
    X = df.drop('binding_affinity', axis=1)
    y = df['binding_affinity']
    
    # Handle missing values
    X = X.fillna(X.mean())
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    return X, y

def train_random_forest(X_train, y_train, n_estimators=100, max_depth=10, random_state=42):
    """Train Random Forest model"""
    print("\nTraining Random Forest...")
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    print("Random Forest training complete")
    return model

def train_xgboost(X_train, y_train, n_estimators=100, max_depth=6, random_state=42):
    """Train XGBoost model"""
    print("\nTraining XGBoost...")
    model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=0.1,
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    print("XGBoost training complete")
    return model

def train_linear_regression(X_train, y_train, scale=True):
    """Train Linear Regression model"""
    print("\nTraining Linear Regression...")
    
    if scale:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        model = LinearRegression()
        model.fit(X_train_scaled, y_train)
        return model, scaler
    else:
        model = LinearRegression()
        model.fit(X_train, y_train)
        return model, None

def train_ridge_regression(X_train, y_train, alpha=1.0, scale=True):
    """Train Ridge Regression model"""
    print("\nTraining Ridge Regression...")
    
    if scale:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        model = Ridge(alpha=alpha)
        model.fit(X_train_scaled, y_train)
        return model, scaler
    else:
        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)
        return model, None

def evaluate_model(model, X_test, y_test, scaler=None, model_name="Model"):
    """Evaluate model performance"""
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    # Prepare test data
    if scaler is not None:
        X_test_scaled = scaler.transform(X_test)
        y_pred = model.predict(X_test_scaled)
    else:
        y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n{model_name} Performance:")
    print(f"  RMSE: {rmse:.2f} nM")
    print(f"  MAE: {mae:.2f} nM")
    print(f"  R²: {r2:.4f}")
    
    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'predictions': y_pred
    }

def cross_validate_model(model, X, y, cv=5, scaler=None, model_name="Model"):
    """Perform cross-validation"""
    print(f"\nCross-validating {model_name}...")
    
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    
    if scaler is not None:
        X_scaled = scaler.fit_transform(X)
        scores = cross_val_score(model, X_scaled, y, cv=kf, 
                               scoring='neg_mean_squared_error', n_jobs=-1)
    else:
        scores = cross_val_score(model, X, y, cv=kf, 
                               scoring='neg_mean_squared_error', n_jobs=-1)
    
    rmse_scores = np.sqrt(-scores)
    
    print(f"  CV RMSE: {rmse_scores.mean():.2f} ± {rmse_scores.std():.2f} nM")
    
    return rmse_scores

def train_all_models(X_train, X_test, y_train, y_test):
    """Train and evaluate all models"""
    print("\n" + "=" * 60)
    print("TRAINING ALL MODELS")
    print("=" * 60)
    
    results = {}
    models = {}
    
    # 1. Random Forest
    rf_model = train_random_forest(X_train, y_train)
    rf_results = evaluate_model(rf_model, X_test, y_test, model_name="Random Forest")
    results['Random Forest'] = rf_results
    models['Random Forest'] = rf_model
    
    # 2. XGBoost
    try:
        xgb_model = train_xgboost(X_train, y_train)
        xgb_results = evaluate_model(xgb_model, X_test, y_test, model_name="XGBoost")
        results['XGBoost'] = xgb_results
        models['XGBoost'] = xgb_model
    except Exception as e:
        print(f"XGBoost training failed: {e}")
    
    # 3. Linear Regression
    lr_model, lr_scaler = train_linear_regression(X_train, y_train, scale=True)
    lr_results = evaluate_model(lr_model, X_test, y_test, scaler=lr_scaler, model_name="Linear Regression")
    results['Linear Regression'] = lr_results
    models['Linear Regression'] = (lr_model, lr_scaler)
    
    # 4. Ridge Regression
    ridge_model, ridge_scaler = train_ridge_regression(X_train, y_train, alpha=1.0, scale=True)
    ridge_results = evaluate_model(ridge_model, X_test, y_test, scaler=ridge_scaler, model_name="Ridge Regression")
    results['Ridge Regression'] = ridge_results
    models['Ridge Regression'] = (ridge_model, ridge_scaler)
    
    return models, results

def save_model(model, filepath, scaler=None):
    """Save trained model"""
    if scaler is not None:
        model_data = {'model': model, 'scaler': scaler}
    else:
        model_data = {'model': model}
    
    with open(filepath, 'wb') as f:
        pickle.dump(model_data, f)
    print(f"Model saved to: {filepath}")

def compare_models(results):
    """Compare and display model performance"""
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    
    comparison = []
    for model_name, metrics in results.items():
        comparison.append({
            'Model': model_name,
            'RMSE (nM)': f"{metrics['rmse']:.2f}",
            'MAE (nM)': f"{metrics['mae']:.2f}",
            'R²': f"{metrics['r2']:.4f}"
        })
    
    df_comparison = pd.DataFrame(comparison)
    print("\n" + df_comparison.to_string(index=False))
    
    # Find best model
    best_model = min(results.items(), key=lambda x: x[1]['rmse'])
    print(f"\nBest Model (lowest RMSE): {best_model[0]}")
    print(f"  RMSE: {best_model[1]['rmse']:.2f} nM")
    print(f"  R²: {best_model[1]['r2']:.4f}")
    
    return df_comparison

if __name__ == "__main__":
    # Load features
    X, y = load_features()
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"\nTrain set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Train all models
    models, results = train_all_models(X_train, X_test, y_train, y_test)
    
    # Compare models
    comparison_df = compare_models(results)
    
    # Save best model
    best_model_name = min(results.items(), key=lambda x: x[1]['rmse'])[0]
    print(f"\nSaving best model: {best_model_name}")
    
    if isinstance(models[best_model_name], tuple):
        model, scaler = models[best_model_name]
        save_model(model, f'models/{best_model_name.lower().replace(" ", "_")}.pkl', scaler)
    else:
        save_model(models[best_model_name], f'models/{best_model_name.lower().replace(" ", "_")}.pkl')
    
    # Save comparison results
    comparison_df.to_csv('results/model_comparison.csv', index=False)
    print("Results saved to: results/model_comparison.csv")

