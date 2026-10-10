"""
Proteus Arc — Comprehensive Notebook Enhancement Script
Injects rich theory, images, mathematical formulas, algorithm descriptions,
and research documentation into the experiment notebook.
"""
import nbformat as nbf
import base64
import os

notebook_path = "notebooks/Proteus_Arc_Real_EEG_Experiments_fixed.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = nbf.read(f, as_version=4)

# ============================================================
# Helper: Create image display code cell
# ============================================================
def make_image_cell(image_path, caption, width=800):
    return nbf.v4.new_code_cell(
        f"from IPython.display import Image, display, HTML\n"
        f"try:\n"
        f"    display(Image(filename='{image_path}', width={width}))\n"
        f"except FileNotFoundError:\n"
        f"    display(HTML('<p style=\"color:gray;\">Image not available in this environment: {image_path}</p>'))\n"
        f"print('{caption}')"
    )

# ============================================================
# SECTION 0: Title Page (replace existing)
# ============================================================
title_cell = nbf.v4.new_markdown_cell("""# Proteus Arc — EEG-Based Alzheimer's Disease Classification

---

## A Comparative Study of Traditional Machine Learning and Deep Learning for EEG-Based Diagnosis

**Author:** Sai Chintamani  
**Dataset:** OpenNeuro ds004504 (Miltiadous et al., 2023)  
**License:** CC0 (Public Domain)  
**Date:** 2026

---

### Abstract

This notebook implements a complete end-to-end pipeline for classifying electroencephalography (EEG) recordings into three diagnostic categories: **Alzheimer's Disease (AD)**, **Frontotemporal Dementia (FTD)**, and **Cognitively Normal (CN)** controls. We systematically compare two approaches:

- **Candidate 1** — Traditional machine learning classifiers (Logistic Regression, Support Vector Machine, Random Forest) operating on hand-crafted time-domain, frequency-domain, and Hjorth features.
- **Candidate 2** — A compact 1D Convolutional Neural Network (CNN) learning representations directly from raw EEG windows.

All experiments enforce strict **participant-level data splitting** to prevent information leakage and ensure that reported metrics reflect genuine generalization to unseen individuals.

### Research Question

> *"Can a compact 1D CNN learn useful Alzheimer's-related patterns directly from EEG signals and generalize to previously unseen participants better than conventional machine-learning models using engineered EEG features?"*

### Key Design Principles

| Principle | Implementation |
|-----------|---------------|
| No data leakage | Participant-level `GroupShuffleSplit` with disjointness assertions |
| No fabricated results | Notebook halts with `FileNotFoundError` if dataset is missing |
| Reproducibility | Fixed random seed (42) across all stochastic operations |
| Class imbalance handling | Balanced class weights in all models |
| Fair comparison | Both candidates evaluated on identical test participants |

> **⚠️ Disclaimer:** This is a research prototype, not a clinically validated diagnostic system. Performance on this dataset alone does not establish clinical diagnostic reliability.
""")

# ============================================================
# SECTION: Pipeline Architecture Overview
# ============================================================
pipeline_overview_md = nbf.v4.new_markdown_cell("""## Pipeline Architecture Overview

The Proteus Arc pipeline consists of six sequential stages, from raw EEG ingestion through to comparative evaluation. The diagram below illustrates the complete data flow.

### Stage Descriptions

| Stage | Description | Output |
|-------|-------------|--------|
| **1. Data Ingestion** | Load OpenNeuro ds004504 derivatives (`.set` files) via MNE-Python | Raw `mne.io.Raw` objects |
| **2. Preprocessing** | Bandpass filter (0.5–45 Hz, 4th-order Butterworth IIR) | Filtered continuous signals |
| **3. Quality Control** | Check for NaN/Inf values and flat channels | Validated signals |
| **4. Windowing** | Segment into 30-second epochs with 50% overlap | `(n_windows, 19, 15000)` arrays |
| **5. Modeling** | Train Candidate 1 (feature-based ML) and Candidate 2 (1D CNN) | Trained models |
| **6. Evaluation** | Window-level and subject-level metrics with confusion matrices | Performance comparison |
""")

pipeline_image_cell = make_image_cell(
    'notebooks/images/pipeline_architecture.png',
    'Figure 1: Proteus Arc — End-to-End Pipeline Architecture',
    900
)

# ============================================================
# SECTION: EEG Background and Montage
# ============================================================
eeg_background_md = nbf.v4.new_markdown_cell("""## Electroencephalography (EEG) Background

### What is EEG?

Electroencephalography (EEG) is a non-invasive neuroimaging technique that measures electrical activity produced by neuronal populations in the cerebral cortex. Voltage fluctuations are recorded from electrodes placed on the scalp surface.

### The International 10-20 System

The dataset uses the **International 10-20 electrode placement system**, a standardized method for describing and applying the location of scalp electrodes. The "10" and "20" refer to the fact that the distances between adjacent electrodes are either 10% or 20% of the total front-to-back or right-to-left distance of the skull.

### Clinical EEG Frequency Bands

EEG oscillations are conventionally decomposed into frequency bands, each associated with distinct cognitive and pathological states:

| Band | Frequency Range | Physiological Association | AD Relevance |
|------|----------------|--------------------------|--------------|
| **Delta (δ)** | 0.5 – 4.0 Hz | Deep sleep, pathological states | ↑ Increased in moderate-severe AD |
| **Theta (θ)** | 4.0 – 8.0 Hz | Drowsiness, memory encoding | ↑ Increased in early AD |
| **Alpha (α)** | 8.0 – 13.0 Hz | Relaxed wakefulness, posterior dominant rhythm | ↓ Decreased in AD (slowing) |
| **Beta (β)** | 13.0 – 30.0 Hz | Active thinking, motor planning | Variable changes |

### EEG Alterations in Alzheimer's Disease

The hallmark EEG finding in AD is a progressive **"slowing"** of the dominant posterior rhythm, characterized by:

1. **Reduced alpha power** — The normal 8–13 Hz posterior dominant rhythm decreases in amplitude and frequency.
2. **Increased theta/delta power** — Pathological slow-wave activity increases, particularly over temporal and parietal regions.
3. **Reduced complexity** — The EEG signal becomes more regular and less complex, reflecting reduced information processing capacity.
4. **Altered coherence** — Functional connectivity between brain regions (measured via coherence) becomes disrupted.

These changes correlate with disease severity as measured by the Mini-Mental State Examination (MMSE).
""")

eeg_montage_image = make_image_cell(
    'notebooks/images/eeg_montage.png',
    'Figure 2: International 10-20 System — 19-Channel EEG Montage Used in ds004504',
    600
)

# ============================================================
# SECTION: Dataset Description
# ============================================================
dataset_md = nbf.v4.new_markdown_cell("""## Dataset: OpenNeuro ds004504

### Citation

> Miltiadous, A., Tzimourta, K.D., Afrantou, T., Ioannidis, P., Grigoriadis, N., Tsalikakis, D.G., Angelidis, P., Tsipouras, M.G., Glavas, E., Giannakeas, N. and Tzallas, A.T. (2023). *A dataset of scalp EEG recordings of Alzheimer's disease, frontotemporal dementia and healthy subjects from routine EEG.* Data, 8(6), p.95.

### Cohort Summary

| Property | Value |
|----------|-------|
| Total participants | 88 |
| Alzheimer's Disease (AD) | 36 participants |
| Frontotemporal Dementia (FTD) | 23 participants |
| Cognitively Normal (CN) | 29 participants |
| EEG system | 19-channel, 10-20 montage |
| Sampling rate | 500 Hz |
| Recording condition | Eyes-closed resting state |
| Preprocessing (by authors) | Band-pass filtering, ASR artifact removal, ICA |
| License | CC0 (Public Domain) |

### Pre-existing Preprocessing

The dataset authors applied:
1. **Band-pass filtering** — Initial frequency-domain cleaning.
2. **Artifact Subspace Reconstruction (ASR)** — Removes high-amplitude transient artifacts.
3. **Independent Component Analysis (ICA)** — Removes stereotyped artifacts (eye blinks, muscle activity, line noise).

> **⚠️ Known Limitation:** ASR and ICA were applied to the continuous recordings before segmentation. This means the artifact rejection models were fitted on the full recording, which includes data that may end up in different splits. This is inherent to the dataset and cannot be corrected without access to the raw, unprocessed recordings. This limitation must be acknowledged in any research report.
""")

# ============================================================
# SECTION: Feature Extraction Theory
# ============================================================
feature_theory_md = nbf.v4.new_markdown_cell(r"""## Feature Engineering Theory (Candidate 1)

Candidate 1 extracts hand-crafted features from each EEG window across three domains. For each of the 19 channels, we compute features independently, resulting in a feature vector of dimensionality $19 \times N_{features}$.

### 1. Time-Domain Features

These features capture the statistical properties of the raw EEG amplitude distribution:

| Feature | Formula | Interpretation |
|---------|---------|----------------|
| **Mean** | $\bar{x} = \frac{1}{N}\sum_{i=1}^{N} x_i$ | DC offset (typically ≈ 0 after filtering) |
| **Standard Deviation** | $\sigma = \sqrt{\frac{1}{N-1}\sum_{i=1}^{N}(x_i - \bar{x})^2}$ | Signal amplitude variability |
| **Variance** | $\sigma^2$ | Mean power of the signal |
| **Root Mean Square** | $\text{RMS} = \sqrt{\frac{1}{N}\sum_{i=1}^{N} x_i^2}$ | Effective signal amplitude |
| **Skewness** | $\gamma_1 = \frac{1}{N}\sum_{i=1}^{N}\left(\frac{x_i - \bar{x}}{\sigma}\right)^3$ | Asymmetry of amplitude distribution |
| **Kurtosis** | $\kappa = \frac{1}{N}\sum_{i=1}^{N}\left(\frac{x_i - \bar{x}}{\sigma}\right)^4 - 3$ | Tail heaviness (transient detection) |

### 2. Frequency-Domain Features (Welch's Method)

We estimate the **Power Spectral Density (PSD)** using Welch's method, which divides the signal into overlapping segments, computes the periodogram of each, and averages them to reduce variance.

For each clinical EEG band $b$ with frequency range $[f_{low}, f_{high}]$:

$$P_b = \int_{f_{low}}^{f_{high}} S_{xx}(f) \, df$$

where $S_{xx}(f)$ is the PSD estimate. We compute **absolute band power** for:

| Band | Range | Clinical Significance in AD |
|------|-------|----------------------------|
| Delta ($\delta$) | 0.5 – 4.0 Hz | Pathological slowing marker |
| Theta ($\theta$) | 4.0 – 8.0 Hz | Early cognitive decline indicator |
| Alpha ($\alpha$) | 8.0 – 13.0 Hz | Posterior dominant rhythm integrity |
| Beta ($\beta$) | 13.0 – 30.0 Hz | Cortical activation level |

### 3. Hjorth Parameters

Hjorth parameters (Hjorth, 1970) provide a compact, computationally efficient description of the signal's temporal structure using only variance computations:

| Parameter | Formula | Interpretation |
|-----------|---------|----------------|
| **Activity** | $H_0 = \text{var}(x)$ | Total signal power |
| **Mobility** | $H_1 = \sqrt{\frac{\text{var}(x')}{\text{var}(x)}}$ | Mean frequency (proportion of standard deviation of the power spectrum) |
| **Complexity** | $H_2 = \frac{H_1(x')}{H_1(x)}$ | Bandwidth — how similar the signal is to a pure sine wave |

where $x'$ denotes the first temporal derivative of the signal.
""")

feature_image_cell = make_image_cell(
    'notebooks/images/feature_extraction.png',
    'Figure 3: Feature Extraction Pipeline — Time Domain, Frequency Domain, and Hjorth Parameters',
    800
)

# ============================================================
# SECTION: Traditional ML Algorithms
# ============================================================
ml_algorithms_md = nbf.v4.new_markdown_cell(r"""## Traditional Machine Learning Classifiers (Candidate 1)

All three classifiers are wrapped in `sklearn.Pipeline` objects containing:
1. **SimpleImputer** (median strategy) — Handles any NaN values from feature extraction.
2. **StandardScaler** — Zero-mean, unit-variance normalization fitted **only on training data**.
3. **Classifier** — The actual classification algorithm.

### 1. Logistic Regression (LR)

A linear model that estimates the posterior probability of each class using the softmax function:

$$P(y=k | \mathbf{x}) = \frac{\exp(\mathbf{w}_k^\top \mathbf{x} + b_k)}{\sum_{j=1}^{K} \exp(\mathbf{w}_j^\top \mathbf{x} + b_j)}$$

- **Purpose:** Establishes a linear separability baseline.
- **Regularization:** L2 penalty (default).
- **Class balancing:** `class_weight='balanced'` — inversely proportional to class frequency.

### 2. Support Vector Machine (SVM)

An SVM with a Radial Basis Function (RBF) kernel maps the features into an infinite-dimensional space to find a maximum-margin hyperplane:

$$K(\mathbf{x}_i, \mathbf{x}_j) = \exp\left(-\gamma \|\mathbf{x}_i - \mathbf{x}_j\|^2\right)$$

- **Purpose:** Tests whether non-linear feature interactions improve classification.
- **Kernel:** RBF with `gamma='scale'`.
- **Probability calibration:** Enabled via Platt scaling for ROC-AUC computation.

### 3. Random Forest (RF)

An ensemble of 300 decision trees, each trained on a bootstrap sample with random feature subsets:

$$\hat{y} = \text{mode}\left\{h_t(\mathbf{x})\right\}_{t=1}^{T}$$

where $h_t$ is the $t$-th decision tree.

- **Purpose:** Captures complex non-linear interactions and provides feature importance rankings.
- **Configuration:** `n_estimators=300`, `min_samples_leaf=2`, `class_weight='balanced'`.
""")

# ============================================================
# SECTION: CNN Architecture
# ============================================================
cnn_theory_md = nbf.v4.new_markdown_cell(r"""## 1D Convolutional Neural Network Architecture (Candidate 2)

### Motivation

Unlike Candidate 1, which relies on hand-crafted features requiring domain expertise, Candidate 2 learns hierarchical spatial-temporal representations **directly from the raw EEG signal**. The 1D CNN applies temporal convolutions across the time axis, treating each EEG channel as a separate input "feature map."

### Architecture Summary

```
Input: (batch_size, 19 channels, 15000 samples)
                    │
    ┌───────────────┴───────────────┐
    │     CONVOLUTIONAL BLOCK 1     │
    │  Conv1D(19→16, kernel=31)     │
    │  BatchNorm1D(16)              │
    │  ReLU                         │
    │  MaxPool1D(4) → 3750 samples  │
    └───────────────┬───────────────┘
                    │
    ┌───────────────┴───────────────┐
    │     CONVOLUTIONAL BLOCK 2     │
    │  Conv1D(16→32, kernel=15)     │
    │  BatchNorm1D(32)              │
    │  ReLU                         │
    │  MaxPool1D(4) → 937 samples   │
    └───────────────┬───────────────┘
                    │
    ┌───────────────┴───────────────┐
    │     GLOBAL POOLING            │
    │  AdaptiveAvgPool1D(1) → 1     │
    └───────────────┬───────────────┘
                    │
    ┌───────────────┴───────────────┐
    │     CLASSIFICATION HEAD       │
    │  Dropout(p=0.5)               │
    │  Linear(32 → 3)              │
    │  Output: [AD, CN, FTD]        │
    └───────────────────────────────┘
```

### Design Rationale

| Design Choice | Rationale |
|--------------|-----------|
| **Large initial kernel (31)** | Captures broad temporal patterns (~62 ms at 500 Hz), corresponding to Delta/Theta oscillation morphology |
| **Two convolutional blocks** | Keeps the model compact to prevent overfitting on 88 participants |
| **Batch Normalization** | Stabilizes training and allows higher learning rates |
| **Adaptive Average Pooling** | Produces a fixed-size feature vector regardless of input length |
| **Dropout (p=0.5)** | Strong regularization critical for small biomedical datasets |
| **Weighted Cross-Entropy Loss** | Addresses the class imbalance (AD:36, CN:29, FTD:23) |

### Training Configuration

| Hyperparameter | Value |
|---------------|-------|
| Optimizer | Adam |
| Learning rate | 0.001 |
| Weight decay (L2) | 0.0001 |
| Batch size | 32 |
| Max epochs | 50 |
| Early stopping patience | 10 epochs (on validation loss) |
| Checkpoint selection | Best validation loss |
| Random seed | 42 |
""")

cnn_image_cell = make_image_cell(
    'notebooks/images/cnn_architecture.png',
    'Figure 4: EEG1DCNN — Compact 1D Convolutional Neural Network Architecture',
    900
)

# ============================================================
# SECTION: Participant Splitting Theory
# ============================================================
splitting_theory_md = nbf.v4.new_markdown_cell(r"""## Participant-Level Splitting and Data Leakage Prevention

### The Data Leakage Problem in EEG

In EEG studies, each participant typically produces many windows (epochs). If we perform a naive random train/test split at the **window level**, windows from the same participant will appear in both the training and test sets. The model may then learn **participant-specific artifacts** (electrode impedance, scalp thickness, habitual muscle tension) rather than **disease-related patterns**.

This is known as **data leakage** and leads to artificially inflated performance metrics that do not reflect real-world generalization.

### Our Solution: GroupShuffleSplit

We use scikit-learn's `GroupShuffleSplit` with participant IDs as the grouping variable. This guarantees that **all windows from a single participant** are assigned to exactly one split.

**Two-Stage Splitting Protocol:**

| Stage | Operation | Result |
|-------|-----------|--------|
| Stage 1 | 80/20 split (train+val vs. test) | Test set: ~18 participants |
| Stage 2 | 75/25 split of remainder (train vs. val) | Train: ~53, Val: ~17 participants |

**Effective proportions:** ~60% Train / ~20% Validation / ~20% Test

### Runtime Assertions

The notebook enforces three explicit assertions after every split:

```python
assert train_participants.isdisjoint(val_participants)   # No train/val overlap
assert train_participants.isdisjoint(test_participants)   # No train/test overlap
assert val_participants.isdisjoint(test_participants)      # No val/test overlap
```

If any assertion fails, the notebook halts immediately with a clear error message identifying the leaked participants.

### Additional Safeguards

| Safeguard | Implementation |
|-----------|---------------|
| Baseline scaling | `StandardScaler` inside `sklearn.Pipeline` — fitted on train only |
| CNN class weights | Computed from `y_train` labels only |
| CNN checkpoint | Selected by **validation loss** only (test data never influences model selection) |
| Baselines use train+val | Appropriate since baselines don't use early stopping |
""")

split_image_cell = make_image_cell(
    'notebooks/images/participant_split.png',
    'Figure 5: Participant-Level Data Splitting — Preventing Information Leakage',
    800
)

# ============================================================
# SECTION: Evaluation Methodology
# ============================================================
eval_theory_md = nbf.v4.new_markdown_cell(r"""## Evaluation Methodology

### Window-Level Metrics

Each model produces predictions for every 30-second EEG window in the test set. We compute:

| Metric | Formula | Purpose |
|--------|---------|---------|
| **Accuracy** | $\frac{TP + TN}{TP + TN + FP + FN}$ | Overall correctness |
| **Balanced Accuracy** | $\frac{1}{K}\sum_{k=1}^{K} \text{Recall}_k$ | Corrects for class imbalance |
| **Macro Precision** | $\frac{1}{K}\sum_{k=1}^{K} \frac{TP_k}{TP_k + FP_k}$ | Average per-class positive predictive value |
| **Macro Recall (Sensitivity)** | $\frac{1}{K}\sum_{k=1}^{K} \frac{TP_k}{TP_k + FN_k}$ | Average per-class detection rate |
| **Macro F1** | $\frac{1}{K}\sum_{k=1}^{K} \frac{2 \cdot P_k \cdot R_k}{P_k + R_k}$ | Harmonic mean of precision and recall |
| **ROC-AUC** | Area under the one-vs-rest ROC curve | Discrimination ability across thresholds |

### Subject-Level Aggregation (Majority Voting)

For clinical relevance, we aggregate window-level predictions to produce a **single diagnosis per participant**:

$$\hat{y}_{\text{subject}} = \text{mode}\left\{\hat{y}_w : w \in \text{windows}(\text{subject})\right\}$$

This majority-voting approach is more clinically meaningful than window-level metrics because a clinician would make one diagnosis per patient, not per epoch.

### Confusion Matrices

We generate confusion matrices for all four models to visualize:
- Which diagnostic categories are most frequently confused.
- Whether AD vs. FTD discrimination is harder than AD vs. CN.
- Class-specific sensitivity and specificity patterns.
""")

# ============================================================
# Now inject all cells into the notebook at the correct positions
# ============================================================
new_cells = []
inserted_title = False
inserted_pipeline = False
inserted_eeg_bg = False
inserted_dataset = False
inserted_features = False
inserted_ml = False
inserted_cnn = False
inserted_split = False
inserted_eval = False

for i, cell in enumerate(nb.cells):
    src = cell.source if hasattr(cell, 'source') else ''

    # Replace the original title cell
    if not inserted_title and cell.cell_type == 'markdown' and '# Proteus Arc' in src and 'Real EEG Experiments' in src:
        new_cells.append(title_cell)
        inserted_title = True
        continue

    # Insert pipeline overview before Environment Inspection
    if not inserted_pipeline and cell.cell_type == 'markdown' and '## 1. Environment Inspection' in src:
        new_cells.append(pipeline_overview_md)
        new_cells.append(pipeline_image_cell)
        inserted_pipeline = True

    # Insert EEG background before Dataset Access
    if not inserted_eeg_bg and cell.cell_type == 'markdown' and '## 4. Dataset Access' in src:
        new_cells.append(eeg_background_md)
        new_cells.append(eeg_montage_image)
        inserted_eeg_bg = True

    # Insert dataset theory before Dataset Audit
    if not inserted_dataset and cell.cell_type == 'markdown' and '## 5. Dataset Audit' in src:
        new_cells.append(dataset_md)
        inserted_dataset = True

    # Insert feature extraction theory before Cell 11
    if not inserted_features and cell.cell_type == 'markdown' and '## 11. Baseline Feature Extraction' in src:
        new_cells.append(feature_theory_md)
        new_cells.append(feature_image_cell)
        inserted_features = True

    # Insert ML algorithms theory before Cell 12
    if not inserted_ml and cell.cell_type == 'markdown' and '## 12. Train Logistic Regression' in src:
        new_cells.append(ml_algorithms_md)
        inserted_ml = True

    # Insert CNN theory before Cell 15
    if not inserted_cnn and cell.cell_type == 'markdown' and '## 15. CNN Initialization' in src:
        new_cells.append(cnn_theory_md)
        new_cells.append(cnn_image_cell)
        inserted_cnn = True

    # Insert splitting theory before Cell 10
    if not inserted_split and cell.cell_type == 'markdown' and '## 10. Participant-Level' in src:
        new_cells.append(splitting_theory_md)
        new_cells.append(split_image_cell)
        inserted_split = True

    # Insert evaluation theory before Cell 17
    if not inserted_eval and cell.cell_type == 'markdown' and '## 17. Final Held-Out Test' in src:
        new_cells.append(eval_theory_md)
        inserted_eval = True

    # Remove previously injected partial cells from earlier injection
    if cell.cell_type == 'markdown' and '## Mathematical Formulation of Engineered Features' in src:
        continue
    if cell.cell_type == 'markdown' and '## Traditional Machine Learning Algorithms (Candidate 1)' in src and 'We evaluate three classical' in src:
        continue
    if cell.cell_type == 'markdown' and '## 1D Convolutional Neural Network (Candidate 2)' in src and 'Unlike Candidate 1' in src:
        continue

    new_cells.append(cell)

nb.cells = new_cells

# Save
with open(notebook_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f"SUCCESS: Notebook enhanced with {len(nb.cells)} cells")
print("Added: Title page, Pipeline overview, EEG background, Dataset theory,")
print("       Feature extraction theory, ML algorithms, CNN architecture,")
print("       Participant splitting theory, Evaluation methodology")
print("Added: 5 scientific diagrams (pipeline, CNN, EEG montage, split, features)")
