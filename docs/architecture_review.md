# Architecture Review

## Existing Implementation Analysis

### Preprocessing and Feature Extraction
The existing notebooks (`Proteus_Arc_ds004504_EEG_Alzheimers.ipynb` and `Proteus_Arc_EEG_Alzheimers_Classification.ipynb`) outline the following data ingestion and preprocessing pipeline:
- **Data Source**: Fetches OpenNeuro ds004504 derivatives using `openneuro-py`.
- **Ingestion**: MNE-Python `read_raw_eeglab` loads `.set` files.
- **Windowing**: Custom epoching slices recordings into 30-second windows with a 50% overlap. 
- **Feature Extraction**: Extracts 380 features per epoch across 19 channels (20 features per channel). The features include time-domain statistics, Hjorth parameters, and frequency-domain Welch PSD band powers (Delta, Theta, Alpha, Beta).

### Machine Learning Baseline (Candidate 1)
- The pipeline establishes **Logistic Regression**, **SVM (RBF kernel)**, and **Random Forest** baselines.
- The pipeline applies `SimpleImputer` (median) and `StandardScaler` (for LR and SVM) correctly within `sklearn.pipeline.Pipeline`, avoiding data leakage from test to train sets.
- `StratifiedGroupKFold` or `LeaveOneGroupOut` is utilized to ensure **participant-independent splits**, avoiding epoch leakage.

### Missing Components (Action Items)
1. **T-GCN Implementation**: The original roadmap mentions a provided T-GCN notebook, but this is **missing** from the workspace. An integration point needs to be explicitly created in the pipeline to allow fair comparison once the graph architecture is established.
2. **1D CNN Implementation (Candidate 2)**: There is no existing 1D CNN implementation in PyTorch. The proposed architecture should accommodate raw EEG (or appropriately preprocessed time-series) as input `(batch_size, channels, samples)`.
3. **Data Localization**: The real data download requires an active OpenNeuro connection. A synthetic data mock is used in the codebase when OpenNeuro is unreachable.

### Summary
The existing codebase sets a strong foundation for participant-independent validation, leakage prevention, and classical feature engineering. However, it lacks deep learning implementations (CNN, T-GCN) required for the Proteus Arc experiments.
