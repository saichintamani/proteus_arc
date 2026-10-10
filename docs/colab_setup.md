# Google Colab Setup Guide — Proteus Arc

## Prerequisites
- A Google account with access to [Google Colab](https://colab.research.google.com).
- The Proteus Arc project files (this repository).

## Step-by-Step Setup

### 1. Upload the Project
**Option A — GitHub Clone** (recommended):
```
!git clone https://github.com/YOUR_USERNAME/proteus_arc.git /content/proteus_arc
```

**Option B — Manual Upload**:
1. Zip the entire `protus aec/` directory on your local machine.
2. In Colab, use the file browser (left sidebar) to upload the `.zip`.
3. Extract:
```python
!unzip -q /content/proteus_arc.zip -d /content/proteus_arc
```

### 2. Install Dependencies
```python
!pip install -q mne openneuro-py scikit-learn pandas scipy pyyaml joblib matplotlib seaborn torch
```

### 3. Set Working Directory
```python
import os, sys
os.chdir('/content/proteus_arc')
sys.path.insert(0, '/content/proteus_arc')
```

### 4. Download the Dataset
The notebook uses `openneuro-py` to download the **derivatives** subset:
```python
!openneuro-py download --dataset ds004504 \
    --target_dir /content/ds004504 \
    --include 'participants.tsv' 'participants.json' 'dataset_description.json' 'derivatives/*'
```
This downloads ~2.5 GB. Estimated time: 10–20 minutes.

### 5. Run the Notebook
Open `notebooks/Proteus_Arc_Real_EEG_Experiments.ipynb` and select **Runtime > Run All**.

## GPU Configuration
Select **Runtime > Change runtime type > T4 GPU** for CNN training.  
The notebook supports CPU execution as well; the CNN training will simply be slower.

## Environment Compatibility
The notebook has been tested for compatibility with:
- Python 3.10+
- PyTorch 2.0+
- MNE-Python 1.0+
- scikit-learn 1.0+

## Local SciPy DLL Issue
The local Windows Anaconda environment has a known `scipy` DLL import error (`ImportError: DLL load failed while importing _ufuncs`). This is caused by a corrupted or incompatible native BLAS/LAPACK installation within Anaconda on Windows and does **not** affect Google Colab. To fix locally:
```bash
conda create -n proteus_arc python=3.10
conda activate proteus_arc
conda install -c conda-forge scipy scikit-learn numpy pandas mne matplotlib seaborn
pip install torch pyyaml joblib openneuro-py
```
