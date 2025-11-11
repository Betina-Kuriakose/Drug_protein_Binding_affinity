"""
Main Pipeline for Drug-Protein Binding Affinity Prediction

This script runs the complete pipeline:
1. Data preprocessing
2. Feature engineering
3. Model training
4. Model evaluation
5. Results visualization
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Add src to path
sys.path.append('src')

# Import modules
from data_preprocessing import preprocess_pipeline
from feature_engineering import combine_features
from models import train_all_models, compare_models, save_model
from evaluation import create_evaluation_report

# Set project directory
project_dir = r"C:\Users\Betina Kuriakose\OneDrive\Desktop\Drug_protein_binding_affinity"
os.chdir(project_dir)

def main():
    """Main pipeline execution"""
    print("=" * 70)
    print("DRUG-PROTEIN BINDING AFFINITY PREDICTION - ML PIPELINE")
    print("=" * 70)
    
    # Step 1: Data Preprocessing
    print("\n" + "=" * 70)
    print("STEP 1: DATA PREPROCESSING")
    print("=" * 70)
    
    X_train, X_test, y_train, y_test, df_processed = preprocess_pipeline(
        filepath='dataset/binding_affinity_data.csv',
        test_size=0.2,
        random_state=42
    )
    
    # Save processed data
    df_processed.to_csv('dataset/processed_data.csv', index=False)
    print("\nProcessed data saved to: dataset/processed_data.csv")
    
    # Step 2: Feature Engineering
    print("\n" + "=" * 70)
    print("STEP 2: FEATURE ENGINEERING")
    print("=" * 70)
    
    # Combine features for train and test separately
    print("\nCreating features for training set...")
    X_train_features = combine_features(X_train, use_fingerprints=True, n_bits=512)
    
    print("\nCreating features for test set...")
    X_test_features = combine_features(X_test, use_fingerprints=True, n_bits=512)
    
    # Ensure same columns
    common_cols = X_train_features.columns.intersection(X_test_features.columns)
    X_train_features = X_train_features[common_cols]
    X_test_features = X_test_features[common_cols]
    
    # Handle missing values
    X_train_features = X_train_features.fillna(X_train_features.mean())
    X_test_features = X_test_features.fillna(X_train_features.mean())
    
    print(f"\nFinal feature matrix shape - Train: {X_train_features.shape}, Test: {X_test_features.shape}")
    
    # Save features
    X_train_features['binding_affinity'] = y_train.values
    X_test_features['binding_affinity'] = y_test.values
    X_train_features.to_csv('dataset/features_train.csv', index=False)
    X_test_features.to_csv('dataset/features_test.csv', index=False)
    print("Features saved to: dataset/features_train.csv and dataset/features_test.csv")
    
    # Step 3: Model Training
    print("\n" + "=" * 70)
    print("STEP 3: MODEL TRAINING")
    print("=" * 70)
    
    models, results = train_all_models(
        X_train_features.drop('binding_affinity', axis=1),
        X_test_features.drop('binding_affinity', axis=1),
        y_train,
        y_test
    )
    
    # Step 4: Model Comparison
    print("\n" + "=" * 70)
    print("STEP 4: MODEL COMPARISON")
    print("=" * 70)
    
    comparison_df = compare_models(results)
    comparison_df.to_csv('results/model_comparison.csv', index=False)
    
    # Step 5: Evaluation and Visualization
    print("\n" + "=" * 70)
    print("STEP 5: MODEL EVALUATION")
    print("=" * 70)
    
    os.makedirs('results', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    for model_name, metrics in results.items():
        print(f"\nGenerating evaluation report for {model_name}...")
        
        # Get predictions
        y_pred = metrics['predictions']
        
        # Create evaluation report
        create_evaluation_report(
            y_test.values,
            y_pred,
            model_name=model_name,
            save_dir='results'
        )
        
        # Save best model
        if model_name == min(results.items(), key=lambda x: x[1]['rmse'])[0]:
            print(f"\nSaving best model: {model_name}")
            model_obj = models[model_name]
            if isinstance(model_obj, tuple):
                model, scaler = model_obj
                save_model(model, f'models/{model_name.lower().replace(" ", "_")}.pkl', scaler)
            else:
                save_model(model_obj, f'models/{model_name.lower().replace(" ", "_")}.pkl')
    
    # Final Summary
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE - SUMMARY")
    print("=" * 70)
    
    best_model = min(results.items(), key=lambda x: x[1]['rmse'])
    print(f"\nBest Model: {best_model[0]}")
    print(f"  RMSE: {best_model[1]['rmse']:.2f} nM")
    print(f"  MAE: {best_model[1]['mae']:.2f} nM")
    print(f"  R²: {best_model[1]['r2']:.4f}")
    
    print("\nFiles Generated:")
    print("  - dataset/processed_data.csv")
    print("  - dataset/features_train.csv")
    print("  - dataset/features_test.csv")
    print("  - results/model_comparison.csv")
    print("  - results/*_metrics.csv")
    print("  - results/*_predictions.png")
    print("  - results/*_residuals.png")
    print("  - models/*.pkl")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nError in pipeline: {e}")
        import traceback
        traceback.print_exc()

