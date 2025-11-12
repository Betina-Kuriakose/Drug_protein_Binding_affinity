import pandas as pd
import numpy as np
import os


# Read the dataset
df = pd.read_csv('dataset/binding_affinity_data.csv')

print("=" * 60)
print("DATASET ANALYSIS FOR DRUG-PROTEIN BINDING AFFINITY RESEARCH")
print("=" * 60)

print(f"\n1. DATASET OVERVIEW")
print(f"   - Total rows: {len(df)}")
print(f"   - Total columns: {len(df.columns)}")
print(f"   - Total missing values: {df.isnull().sum().sum()}")

print(f"\n2. KEY FEATURES FOR ML MODEL")
print(f"   - Ligand SMILES: {df['Ligand SMILES'].notna().sum()} non-null values")
print(f"   - Target sequences: {df['BindingDB Target Chain Sequence'].notna().sum()} non-null values")

print(f"\n3. BINDING AFFINITY VALUES (Target Variable)")
affinity_cols = ['Ki (nM)', 'IC50 (nM)', 'Kd (nM)', 'EC50 (nM)']
for col in affinity_cols:
    non_null = df[col].notna().sum()
    if non_null > 0:
        # Convert to numeric, coerce errors to NaN
        numeric_col = pd.to_numeric(df[col], errors='coerce')
        valid_values = numeric_col.dropna()
        if len(valid_values) > 0:
            print(f"   - {col}: {len(valid_values)} numeric values")
            print(f"     Range: {valid_values.min():.2f} - {valid_values.max():.2f} nM")
        else:
            print(f"   - {col}: {non_null} values (non-numeric)")

# Check which affinity measure has most data
best_affinity = None
max_count = 0
for col in affinity_cols:
    numeric_col = pd.to_numeric(df[col], errors='coerce')
    count = numeric_col.notna().sum()
    if count > max_count:
        max_count = count
        best_affinity = col

print(f"\n   -> Best affinity measure: {best_affinity} ({max_count} values)")

print(f"\n4. PROTEIN INFORMATION")
print(f"   - Unique targets: {df['Target Name'].nunique()}")
print(f"   - Target with sequence: {df['BindingDB Target Chain Sequence'].notna().sum()}")
print(f"   - PDB IDs available: {df['PDB ID(s) for Ligand-Target Complex'].notna().sum()}")

print(f"\n5. DRUG INFORMATION")
print(f"   - Unique ligands: {df['Ligand SMILES'].nunique()}")
print(f"   - PubChem CIDs: {df['PubChem CID'].notna().sum()}")
print(f"   - ChEMBL IDs: {df['ChEMBL ID of Ligand'].notna().sum()}")

print(f"\n6. DATA QUALITY ASSESSMENT")
# Check for complete records (has both SMILES and affinity)
if best_affinity:
    numeric_affinity = pd.to_numeric(df[best_affinity], errors='coerce')
    complete_records = df[(df['Ligand SMILES'].notna()) & (numeric_affinity.notna())]
    print(f"   - Complete records (SMILES + {best_affinity}): {len(complete_records)}")
    
    # Check for records with both drug and protein info
    complete_with_protein = df[(df['Ligand SMILES'].notna()) & 
                               (numeric_affinity.notna()) & 
                               (df['BindingDB Target Chain Sequence'].notna())]
    print(f"   - Complete records with protein sequence: {len(complete_with_protein)}")
else:
    complete_records = df[df['Ligand SMILES'].notna()]
    complete_with_protein = df[(df['Ligand SMILES'].notna()) & 
                               (df['BindingDB Target Chain Sequence'].notna())]
    print(f"   - Complete records (SMILES only): {len(complete_records)}")
    print(f"   - Complete records with protein sequence: {len(complete_with_protein)}")

print(f"\n7. RECOMMENDATIONS")
if len(complete_records) < 100:
    print("   WARNING: Dataset is quite small (<100 complete records)")
    print("      Consider: Getting more data or using data augmentation")
else:
    print(f"   OK: Dataset size is reasonable ({len(complete_records)} complete records)")

if len(complete_with_protein) < len(complete_records) * 0.5:
    print("   WARNING: Many records missing protein sequences")
    print("      Consider: Using protein names/IDs as features or fetching sequences")
else:
    print(f"   OK: Good protein sequence coverage ({len(complete_with_protein)} records)")

print(f"\n8. SUITABILITY FOR RESEARCH")
print(f"   + Has drug structures (SMILES)")
if best_affinity:
    print(f"   + Has binding affinity values ({best_affinity})")
else:
    print(f"   - Missing binding affinity values")
if len(complete_with_protein) > 50:
    print(f"   + Has protein sequences for most records")
else:
    print(f"   - Limited protein sequence data")

print("\n" + "=" * 60)

