# AI-Powered Prediction of Drug-Protein Binding Affinity Using Machine Learning

## Project Overview

This project aims to develop a machine learning model to predict the binding affinity between drug-like compounds and proteins. Using public datasets and established machine learning architectures, the model will help virtually screen drug candidates, accelerating early-stage drug discovery and reducing the need for expensive lab experiments.

## Objectives

- To build a model that predicts binding strengths of drugs to proteins
- To use public datasets for training and evaluation
- To validate AI's utility for drug discovery

## Dataset Information

### Source
- **Dataset**: BindingDB (Binding Database)
- **Format**: CSV (converted from TSV)
- **Location**: `dataset/binding_affinity_data.csv`

### Dataset Statistics
- **Total Records**: 391 rows
- **Complete Records**: 294 (with both SMILES and binding affinity)
- **Features**: 52 columns
- **Target Variable**: Ki (nM) - 294 numeric values
- **Range**: 0.07 - 281,838.29 nM

### Key Features
- **Drug Information**:
  - Ligand SMILES strings (391 records)
  - PubChem CIDs (391 records)
  - ChEMBL IDs (180 records)
  - Unique ligands: 213

- **Protein Information**:
  - Target sequences (391 records)
  - Unique targets: 2
  - PDB IDs available: 50

- **Binding Affinity**:
  - Ki (nM): 294 values (primary target)
  - Kd (nM): 45 values
  - IC50 (nM): 5 values

## Methodology

### Approach
1. **Data Preprocessing**
   - Clean and prepare SMILES strings
   - Process protein sequences
   - Handle missing values
   - Feature engineering

2. **Molecular Representation**
   - SMILES to molecular descriptors
   - Fingerprint generation
   - Graph-based representations (if needed)

3. **Protein Representation**
   - Sequence encoding
   - Feature extraction from protein sequences

4. **Model Development**
   - Start with simpler models (Random Forest, XGBoost)
   - Evaluate performance
   - Compare different approaches

5. **Evaluation**
   - Cross-validation
   - Performance metrics (RMSE, MAE, R²)
   - Visualization of results

## Project Structure

```
Drug_protein_binding_affinity/
│
├── dataset/
│   ├── binding_affinity_data.csv      # Main dataset (CSV format)
│   └── 4343106D673CD682131D3EBA49C069D1ki.tsv  # Original TSV file
│
├── src/                                # Source code (to be created)
│   ├── data_preprocessing.py           # Data cleaning and preparation
│   ├── feature_engineering.py         # Feature extraction
│   ├── models.py                      # ML model implementations
│   └── evaluation.py                  # Model evaluation metrics
│
├── notebooks/                          # Jupyter notebooks (to be created)
│   └── exploratory_data_analysis.ipynb
│
├── models/                             # Saved models (to be created)
│
├── results/                            # Results and visualizations (to be created)
│
├── convert_tsv_to_csv.py               # Dataset conversion script
├── analyze_dataset.py                  # Dataset analysis script
└── README.md                           # This file
```

## Software & Hardware Requirements

### Software
- **Python**: 3.x
- **Key Libraries**:
  - Pandas, NumPy (data manipulation)
  - Scikit-learn (machine learning)
  - XGBoost (gradient boosting)
  - RDKit (molecular processing)
  - Matplotlib, Seaborn (visualization)

### Hardware
- Standard desktop/laptop
- GPU optional (for future deep learning models)
- Minimum 4GB RAM recommended

## Installation

1. Clone or download this repository

2. Install required packages:
```bash
pip install pandas numpy scikit-learn xgboost rdkit matplotlib seaborn
```

Note: RDKit installation may require additional steps. See [RDKit Installation Guide](https://www.rdkit.org/docs/Install.html)

## Usage

### 1. Dataset Analysis
```bash
python analyze_dataset.py
```

### 2. Data Preprocessing
```bash
python src/data_preprocessing.py
```

### 3. Model Training
```bash
python src/models.py
```

## Current Status

- ✅ Dataset converted from TSV to CSV
- ✅ Dataset analysis completed
- ✅ Project structure defined
- 🔄 Data preprocessing (in progress)
- ⏳ Model development (planned)
- ⏳ Model evaluation (planned)

## Expected Outcomes

The project is expected to produce:
1. A trained machine learning model that accurately predicts drug-protein binding affinity
2. Performance metrics and visualizations
3. Documentation of the methodology and results
4. Demonstration of how AI methods can improve the speed and effectiveness of early drug screening

## Limitations & Considerations

- **Small Dataset**: 294 complete records is relatively small for deep learning
  - Solution: Start with simpler models, use data augmentation if needed
- **Limited Target Diversity**: Only 2 unique protein targets
  - Solution: Focus on these specific proteins or expand dataset
- **Missing Values**: Some features have missing data
  - Solution: Handle appropriately during preprocessing

## References

1. Blanco-González, A. et al., "The Role of AI in Drug Discovery: Challenges and Opportunities." PMC, 2023.
2. Jumper, J. et al., "Highly accurate protein structure prediction with AlphaFold," Nature, 2021.
3. BindingDB Database: https://www.bindingdb.org/
4. RDKit Documentation: https://www.rdkit.org/

## Author

Department of CSE (Data Science)

## License

This project is for academic/research purposes.

---

**Note**: This is a computational research project only; no lab work involved.
