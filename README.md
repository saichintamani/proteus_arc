# Proteus Arc: EEG-based Alzheimer's Disease Classification

![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)
![MNE](https://img.shields.io/badge/MNE_Python-1.6%2B-lightgrey.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

**Proteus Arc** is a research-oriented machine learning pipeline designed to investigate whether a compact 1D Convolutional Neural Network (CNN) can learn useful Alzheimer's-related patterns directly from raw EEG signals, and generalize to unseen participants better than traditional machine-learning models using engineered features.

This repository contains the full end-to-end pipeline—from raw EEG ingestion and preprocessing to feature extraction, participant-level cross-validation, and model comparison.

---

## 🧠 Overview

Alzheimer's Disease (AD) and Frontotemporal Dementia (FTD) alter the brain's electrical activity. This project utilizes the **OpenNeuro ds004504** dataset to classify EEG recordings into three diagnostic categories:
1. **Alzheimer's Disease (AD)**
2. **Frontotemporal Dementia (FTD)**
3. **Cognitively Normal (CN)**

### Research Question
*"Can a compact 1D CNN learn useful diagnostic patterns directly from EEG signals and generalize to previously unseen participants better than conventional machine-learning models using engineered EEG features?"*

---

## 📊 Model Comparison & Architecture

The project systematically compares traditional feature-engineering approaches (Candidate 1) against deep learning approaches (Candidate 2).

### Candidate 1: Traditional Machine Learning
We extract Time-domain, Frequency-domain (Welch's method), and Hjorth parameters from the EEG signals.
* Logistic Regression
* Support Vector Machine (SVM)
* Random Forest

### Candidate 2: Deep Learning (1D CNN)
A compact, 1D Convolutional Neural Network designed to operate directly on windowed EEG signals without manual feature extraction.

*(Include your CNN architecture diagram here)*

---

## 🛠️ Project Structure

```text
proteus_arc/
├── configs/                 # YAML configuration files
│   ├── baseline.yaml        # Traditional ML feature pipeline config
│   ├── cnn.yaml             # 1D CNN architecture and training config
│   ├── data.yaml            # Dataset metadata and channel mapping
│   ├── evaluation.yaml      # Metrics configuration
│   └── preprocessing.yaml   # Filtering and windowing parameters
├── docs/                    # Project documentation and audit logs
├── notebooks/
│   └── Proteus_Arc_Real_EEG_Experiments.ipynb # Main execution notebook
├── scripts/                 # Utility scripts
├── src/proteus_arc/         # Core Python package
│   ├── data/                # Data loaders (MNE integration)
│   ├── evaluation/          # Participant-level splitters and metrics
│   ├── features/            # Feature extraction pipelines
│   ├── models/              # Model definitions (CNN and Baselines)
│   └── preprocessing/       # Signal filtering and windowing
├── tests/                   # Extensive Unit test suite (23 tests)
└── README.md
```

---

## 🚀 Getting Started

### 1. Local Setup
Clone the repository and install the dependencies:
```bash
git clone https://github.com/yourusername/proteus_arc.git
cd proteus_arc
pip install -r requirements.txt
```

### 2. Dataset
This project uses the publicly available OpenNeuro dataset **ds004504**. 
The notebook automatically downloads the required derivatives (preprocessed `.set` files) using `openneuro-py`.

### 3. Execution (Google Colab / Kaggle)
The easiest way to reproduce the results is to run the provided notebook in a cloud environment with a GPU.
1. Upload `notebooks/Proteus_Arc_Real_EEG_Experiments.ipynb` to Colab or Kaggle.
2. Upload this repository as a dataset (Kaggle) or clone it directly in the first cell.
3. Run all cells. The notebook will automatically download the EEG data, train all 4 models, generate plots, and produce a downloadable ZIP of the results.

---

## 🔬 Methodology & Data Leakage Prevention

A critical component of this project is the strict prevention of data leakage. 
* **Participant-Level Splitting:** We use a two-stage `GroupShuffleSplit` to ensure that no participant's data appears in both the training and test sets. 
* **Isolated Scaling:** All feature scaling and imputation are performed inside `sklearn` Pipelines, fitted only on the training set.
* **Subject-Level Voting:** Final evaluations are conducted by aggregating window-level predictions using majority voting per participant.

---

## 📈 Results & Visualizations

*(After running the notebook, upload the generated figures from `outputs/figures/` to your repository and link them here.)*

### Cohort Demographics
![Cohort Demographics](docs/images/placeholder_demographics.png)
*Distribution of Age, MMSE scores, and Diagnostic groups.*

### Model Performance (Subject-Level)
| Model | Accuracy | Balanced Accuracy | Macro F1 |
|-------|----------|-------------------|----------|
| Logistic Regression | *TBD* | *TBD* | *TBD* |
| SVM | *TBD* | *TBD* | *TBD* |
| Random Forest | *TBD* | *TBD* | *TBD* |
| **1D CNN** | **TBD** | **TBD** | **TBD** |

### Confusion Matrices
![Confusion Matrices](docs/images/placeholder_confusion.png)
*Comparison of model predictions across the three diagnostic classes.*

---

## 📝 License
This project is licensed under the MIT License - see the LICENSE file for details. The OpenNeuro ds004504 dataset is distributed under the CC0 license.
