# Project Inventory

## Workspace & Repository
- **Git Repository Root**: `d:\My projects\protus aec` (Note: Git is currently not initialized).
- **Current Branch**: N/A

## Existing Notebooks
- `Proteus_Arc_ds004504_EEG_Alzheimers.ipynb`: Implements conventional ML (Logistic Regression, SVM, Random Forest) with subject-independent validation. Downloads ds004504 dataset.
- `Proteus_Arc_EEG_Alzheimers_Classification.ipynb`: Similar to the above, structured as a clean pipeline, includes a synthetic data generator if the dataset is not available locally.

## Existing Implementations
- **T-GCN Implementation**: **MISSING**. There is no T-GCN notebook or code present in the workspace despite references in the roadmap. A clean integration point will be created.
- **CNN Implementation (Candidate 2)**: **MISSING**. A 1D CNN needs to be implemented as per the roadmap.
- **EEG Preprocessing Code**: Present inside the notebooks. Uses MNE-Python and `scipy.signal` for extracting 380 features per epoch (time-domain, Hjorth, frequency-domain).

## Data Directories
- No local data directories exist. The notebooks are configured to download the `derivatives` folder from OpenNeuro (`ds004504`). 

## Research Documents
- `Proteus_Arc_EEG_Documentation.pdf`: Details the dataset selection, weaknesses, evaluation strategy, and literature basis.
- `Proteus_Arc_EEG_TGCN_ML_Implementation_Roadmap_Sai_Chintamani.pdf`: Outlines the ML workflow, dataset audit checklist, graph construction, and fair comparison rules.
