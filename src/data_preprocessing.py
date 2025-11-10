"""
Data Preprocessing for Drug-Protein Binding Affinity Prediction

This script handles:
- Loading and cleaning the dataset
- Processing SMILES strings
- Processing protein sequences
- Handling missing values
- Creating train/test splits
"""

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

# Set project directory
project_dir = r"C:\Users\Betina Kuriakose\OneDrive\Desktop\Drug_protein_binding_affinity"
os.chdir(project_dir)

def load_data(filepath='dataset/binding_affinity_data.csv'):
    """Load the binding affinity dataset"""
    print("Loading dataset...")
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} rows and {len(df.columns)} columns")
    return df

def clean_binding_affinity(df):
    """Clean and prepare binding affinity values"""
    print("\nCleaning binding affinity data...")
    
    # Convert Ki to numeric, handling errors
    df['Ki (nM)'] = pd.to_numeric(df['Ki (nM)'], errors='coerce')
    
    # Filter rows with valid Ki values
    df_clean = df[df['Ki (nM)'].notna()].copy()
    
    print(f"Records with valid Ki values: {len(df_clean)}")
    print(f"Ki range: {df_clean['Ki (nM)'].min():.2f} - {df_clean['Ki (nM)'].max():.2f} nM")
    
    return df_clean

def validate_smiles(df):
    """Validate and clean SMILES strings"""
    print("\nValidating SMILES strings...")
    
    # Check for missing SMILES
    missing_smiles = df['Ligand SMILES'].isna().sum()
    if missing_smiles > 0:
        print(f"Warning: {missing_smiles} records have missing SMILES")
        df = df[df['Ligand SMILES'].notna()].copy()
    
    # Basic SMILES validation (non-empty strings)
    df = df[df['Ligand SMILES'].str.strip() != ''].copy()
    
    print(f"Valid SMILES records: {len(df)}")
    
    return df

def validate_protein_sequences(df):
    """Validate protein sequences"""
    print("\nValidating protein sequences...")
    
    # Check for missing sequences
    missing_seq = df['BindingDB Target Chain Sequence'].isna().sum()
    if missing_seq > 0:
        print(f"Warning: {missing_seq} records have missing protein sequences")
        df = df[df['BindingDB Target Chain Sequence'].notna()].copy()
    
    # Filter out empty sequences
    df = df[df['BindingDB Target Chain Sequence'].str.strip() != ''].copy()
    
    print(f"Valid protein sequence records: {len(df)}")
    
    return df

def prepare_features(df):
    """Prepare feature columns for modeling"""
    print("\nPreparing features...")
    
    # Select key features
    features = {
        'smiles': 'Ligand SMILES',
        'protein_sequence': 'BindingDB Target Chain Sequence',
        'target_name': 'Target Name',
        'pubchem_cid': 'PubChem CID',
        'chembl_id': 'ChEMBL ID of Ligand'
    }
    
    # Create a clean dataframe with selected features
    df_features = pd.DataFrame()
    for new_name, old_name in features.items():
        if old_name in df.columns:
            df_features[new_name] = df[old_name]
        else:
            print(f"Warning: Column '{old_name}' not found")
            df_features[new_name] = None
    
    # Add target variable
    df_features['binding_affinity'] = df['Ki (nM)'].values
    
    return df_features

def create_train_test_split(df, test_size=0.2, random_state=42):
    """Create train/test split"""
    print(f"\nCreating train/test split (test_size={test_size})...")
    
    X = df.drop('binding_affinity', axis=1)
    y = df['binding_affinity']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=True
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    return X_train, X_test, y_train, y_test

def preprocess_pipeline(filepath='dataset/binding_affinity_data.csv', 
                       test_size=0.2, random_state=42):
    """
    Complete preprocessing pipeline
    
    Returns:
        X_train, X_test, y_train, y_test: Train/test splits
        df_processed: Processed dataframe
    """
    print("=" * 60)
    print("DATA PREPROCESSING PIPELINE")
    print("=" * 60)
    
    # Load data
    df = load_data(filepath)
    
    # Clean binding affinity
    df = clean_binding_affinity(df)
    
    # Validate SMILES
    df = validate_smiles(df)
    
    # Validate protein sequences
    df = validate_protein_sequences(df)
    
    # Prepare features
    df_processed = prepare_features(df)
    
    # Create train/test split
    X_train, X_test, y_train, y_test = create_train_test_split(
        df_processed, test_size=test_size, random_state=random_state
    )
    
    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)
    
    return X_train, X_test, y_train, y_test, df_processed

if __name__ == "__main__":
    # Run preprocessing pipeline
    X_train, X_test, y_train, y_test, df_processed = preprocess_pipeline()
    
    # Save processed data
    print("\nSaving processed data...")
    df_processed.to_csv('dataset/processed_data.csv', index=False)
    print("Saved to: dataset/processed_data.csv")
    
    print(f"\nFinal dataset shape: {df_processed.shape}")
    print(f"Features: {list(df_processed.columns)}")

