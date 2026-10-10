# Experiment Protocol

## 1. Research Question
"Can a compact 1D CNN learn useful Alzheimer's-related patterns directly from EEG signals and generalize to unseen participants better than conventional machine-learning models using engineered EEG features?"

## 2. Dataset and Quality Control
- **Dataset**: OpenNeuro ds004504 (preprocessed derivatives).
- **Inclusion Criteria**: Patients diagnosed with Alzheimer's (AD), Frontotemporal Dementia (FTD), and Healthy Controls (CN).
- **Task**: The default is AD vs CN.

## 3. Preprocessing & Windowing
- **Preprocessing steps**: 
  - Band-pass filtering (0.5 - 45 Hz).
  - ASR (Artifact Subspace Reconstruction).
  - ICA (Independent Component Analysis) with ICLabel removal of eye/jaw components.
- **Window generation**: 30 seconds duration, 50% overlap.

## 4. Partitions and Leakage Prevention
- **Methodology**: Participant-independent validation.
- **Implementation**: `StratifiedGroupKFold` or `Leave-One-Subject-Out`.
- **Constraint**: No overlapping participant IDs between training and testing folds. Preprocessing normalization and imputation are strictly fit on the training data.

## 5. Candidate 1: Conventional ML
- **Models**: Logistic Regression, SVM (RBF), Random Forest.
- **Input**: 380 engineered features per epoch (time, frequency, Hjorth parameters).
- **Evaluation**: Trained via cross-validation; epoch-level and subject-level metrics reported.

## 6. Candidate 2: 1D CNN
- **Architecture**: A compact 1D Convolutional Neural Network.
  - Input: `(batch_size, n_channels, n_samples)`.
  - Convolutional Blocks: Conv1d, BatchNorm1d, ReLU, MaxPool1d.
  - Global Adaptive Pooling, Dropout, and Fully Connected Classifier.
- **Training**: PyTorch, mini-batch training, cross-entropy loss, validated using early stopping on a participant-independent validation set.

## 7. Model Comparison and Metrics
- Models will be evaluated identically on:
  - Accuracy & Balanced Accuracy
  - Macro Precision, Recall, F1-Score
  - Sensitivity and Specificity
  - ROC-AUC
  - Confusion Matrix
- **Subject-level voting**: Both epoch-level and subject-level (majority vote) predictions will be reported.
