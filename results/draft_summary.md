# Draft Summary: AI-Powered Prediction of Drug-Protein Binding Affinity

## Results
- Dataset: 391 records (BindingDB); 294 usable with Ki (nM), SMILES, and protein sequence
- Train/Test: 80/20 split (235/59)
- Features: 512-bit Morgan fingerprints + protein sequence features (amino-acid composition)
- Models evaluated:
  - Random Forest: RMSE ≈ 16,948 nM, MAE ≈ 5,771 nM, R² ≈ 0.171 (best)
  - XGBoost: RMSE ≈ 18,700 nM, MAE ≈ 5,680 nM, R² ≈ -0.010
  - Linear Regression: RMSE ≈ 20,617 nM, MAE ≈ 9,119 nM, R² ≈ -0.228
  - Ridge Regression: RMSE ≈ 20,361 nM, MAE ≈ 8,984 nM, R² ≈ -0.197
- Artifacts saved:
  - Model comparison: `results/model_comparison.csv`
  - Per-model metrics: `results/*_metrics.csv`
  - Plots: `results/*_predictions.png`, `results/*_residuals.png`
  - Best model: `models/random_forest.pkl`

## Discussion
- Random Forest achieved the lowest RMSE and positive R², indicating modest explanatory power compared to linear baselines on this small, heterogeneous dataset.
- Wide dynamic range of Ki (0.07–281,838 nM) and skewed distribution make absolute error metrics large and modeling harder without transformation.
- Fingerprints capture molecular structure well; simple protein sequence features provided additional signal despite limited target diversity (only 2 targets).

## Limitations
- Small dataset size (n=294 complete records) limits generalization and favors simpler models.
- Limited target diversity (2 proteins) restricts model applicability across proteins.
- No log-transform of Ki applied in this baseline; raw-scale optimization emphasizes large values.
- Basic protein features; no pretrained protein embeddings (e.g., ESM, ProtBERT).

## Conclusion
This baseline demonstrates a reproducible ML pipeline for binding affinity prediction using public data. Random Forest performed best among simple models, establishing a solid starting point for further improvements (e.g., log(Ki), hyperparameter tuning, richer molecular/protein embeddings).

## Recommended Next Steps (for final submission)
- Apply log-transform to Ki and re-train/evaluate.
- Tune Random Forest/XGBoost (n_estimators, depth, learning rate).
- Add richer protein features (pretrained embeddings) and evaluate gains.
- Expand dataset with additional targets/compounds where possible.
