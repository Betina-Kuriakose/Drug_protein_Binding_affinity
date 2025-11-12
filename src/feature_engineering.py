"""
Feature Engineering for Drug-Protein Binding Affinity Prediction

This script handles:
- Molecular fingerprint generation from SMILES
- Protein sequence feature extraction
- Combining drug and protein features
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit import DataStructs
import os


def smiles_to_mol(smiles):
    """Convert SMILES string to RDKit molecule object"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol
    except:
        return None

def get_molecular_descriptors(smiles):
    """
    Extract molecular descriptors from SMILES
    
    Returns a dictionary of molecular descriptors
    """
    mol = smiles_to_mol(smiles)
    if mol is None:
        return None
    
    descriptors = {
        'mol_weight': Descriptors.MolWt(mol),
        'logp': Descriptors.MolLogP(mol),
        'num_atoms': mol.GetNumAtoms(),
        'num_bonds': mol.GetNumBonds(),
        'num_rings': rdMolDescriptors.CalcNumRings(mol),
        'num_aromatic_rings': rdMolDescriptors.CalcNumAromaticRings(mol),
        'num_heteroatoms': rdMolDescriptors.CalcNumHeteroatoms(mol),
        'num_rotatable_bonds': rdMolDescriptors.CalcNumRotatableBonds(mol),
        'tpsa': Descriptors.TPSA(mol),  # Topological Polar Surface Area
        'num_hbd': Descriptors.NumHDonors(mol),  # Hydrogen bond donors
        'num_hba': Descriptors.NumHAcceptors(mol),  # Hydrogen bond acceptors
        'fraction_csp3': rdMolDescriptors.CalcFractionCsp3(mol),
    }
    
    return descriptors

def get_morgan_fingerprint(smiles, radius=2, n_bits=2048):
    """
    Generate Morgan fingerprint from SMILES
    
    Args:
        smiles: SMILES string
        radius: Fingerprint radius
        n_bits: Number of bits in fingerprint
    
    Returns:
        numpy array of fingerprint bits
    """
    mol = smiles_to_mol(smiles)
    if mol is None:
        return np.zeros(n_bits)
    
    try:
        fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        arr = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr
    except:
        return np.zeros(n_bits)

def get_protein_features(sequence):
    """
    Extract basic features from protein sequence
    
    Args:
        sequence: Protein sequence string
    
    Returns:
        Dictionary of protein features
    """
    if pd.isna(sequence) or sequence == '':
        return None
    
    sequence = str(sequence).upper()
    
    # Amino acid composition
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    aa_counts = {aa: sequence.count(aa) for aa in amino_acids}
    total_aa = len(sequence)
    
    if total_aa == 0:
        return None
    
    # Calculate frequencies
    aa_freq = {f'{aa}_freq': count / total_aa for aa, count in aa_counts.items()}
    
    features = {
        'sequence_length': total_aa,
        'num_charged_residues': sum(sequence.count(aa) for aa in 'DEKRH'),
        'num_aromatic_residues': sum(sequence.count(aa) for aa in 'FWY'),
        'num_polar_residues': sum(sequence.count(aa) for aa in 'NQSTY'),
        'num_hydrophobic_residues': sum(sequence.count(aa) for aa in 'AILMFWV'),
    }
    
    # Add amino acid frequencies
    features.update(aa_freq)
    
    return features

def create_molecular_features(df):
    """Create molecular features from SMILES"""
    print("Creating molecular features from SMILES...")
    
    # Get molecular descriptors
    descriptor_list = []
    for idx, smiles in enumerate(df['smiles']):
        if idx % 50 == 0:
            print(f"  Processing {idx}/{len(df)}...")
        
        desc = get_molecular_descriptors(smiles)
        if desc is None:
            # Create default descriptors with NaN
            desc = {key: np.nan for key in [
                'mol_weight', 'logp', 'num_atoms', 'num_bonds', 'num_rings',
                'num_aromatic_rings', 'num_heteroatoms', 'num_rotatable_bonds',
                'tpsa', 'num_hbd', 'num_hba', 'fraction_csp3'
            ]}
        descriptor_list.append(desc)
    
    df_descriptors = pd.DataFrame(descriptor_list)
    
    # Add prefix to column names
    df_descriptors.columns = ['mol_' + col for col in df_descriptors.columns]
    
    print(f"Created {len(df_descriptors.columns)} molecular descriptor features")
    
    return df_descriptors

def create_fingerprint_features(df, n_bits=2048):
    """Create molecular fingerprint features"""
    print(f"Creating molecular fingerprints (n_bits={n_bits})...")
    
    fingerprint_list = []
    for idx, smiles in enumerate(df['smiles']):
        if idx % 50 == 0:
            print(f"  Processing {idx}/{len(df)}...")
        
        fp = get_morgan_fingerprint(smiles, radius=2, n_bits=n_bits)
        fingerprint_list.append(fp)
    
    df_fingerprints = pd.DataFrame(fingerprint_list)
    df_fingerprints.columns = [f'fp_{i}' for i in range(n_bits)]
    
    print(f"Created {len(df_fingerprints.columns)} fingerprint features")
    
    return df_fingerprints

def create_protein_features(df):
    """Create protein features from sequences"""
    print("Creating protein features from sequences...")
    
    feature_list = []
    for idx, seq in enumerate(df['protein_sequence']):
        if idx % 50 == 0:
            print(f"  Processing {idx}/{len(df)}...")
        
        features = get_protein_features(seq)
        if features is None:
            # Create default features with NaN
            features = {'sequence_length': np.nan, 'num_charged_residues': np.nan,
                       'num_aromatic_residues': np.nan, 'num_polar_residues': np.nan,
                       'num_hydrophobic_residues': np.nan}
            # Add amino acid frequencies
            amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
            for aa in amino_acids:
                features[f'{aa}_freq'] = np.nan
        
        feature_list.append(features)
    
    df_protein = pd.DataFrame(feature_list)
    
    # Add prefix to column names
    df_protein.columns = ['prot_' + col for col in df_protein.columns]
    
    print(f"Created {len(df_protein.columns)} protein features")
    
    return df_protein

def combine_features(df, use_fingerprints=True, n_bits=2048):
    """
    Combine all features into a single feature matrix
    
    Args:
        df: DataFrame with 'smiles' and 'protein_sequence' columns
        use_fingerprints: Whether to use fingerprints (True) or descriptors (False)
        n_bits: Number of fingerprint bits (if use_fingerprints=True)
    
    Returns:
        Feature matrix as DataFrame
    """
    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING")
    print("=" * 60)
    
    # Create molecular features
    if use_fingerprints:
        df_mol = create_fingerprint_features(df, n_bits=n_bits)
    else:
        df_mol = create_molecular_features(df)
    
    # Create protein features
    df_protein = create_protein_features(df)
    
    # Combine all features
    df_features = pd.concat([df_mol, df_protein], axis=1)
    
    print(f"\nTotal features created: {len(df_features.columns)}")
    print(f"Feature matrix shape: {df_features.shape}")
    
    return df_features

if __name__ == "__main__":
    # Load processed data
    print("Loading processed data...")
    df = pd.read_csv('dataset/processed_data.csv')
    
    # Create features
    df_features = combine_features(df, use_fingerprints=True, n_bits=512)  # Using smaller fingerprints for testing
    
    # Add target variable
    df_features['binding_affinity'] = df['binding_affinity'].values
    
    # Save features
    print("\nSaving features...")
    df_features.to_csv('dataset/features.csv', index=False)
    print("Saved to: dataset/features.csv")

