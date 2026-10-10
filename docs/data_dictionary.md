# Data Dictionary

## Dataset: OpenNeuro ds004504 (Derivatives)

### Participant Metadata (`participants.tsv`)
- `participant_id`: String (e.g., `sub-001`), Unique identifier for the subject.
- `Gender`: Categorical (`M`, `F`).
- `Age`: Integer, Participant age in years.
- `Group`: Categorical (`A` = Alzheimer's Disease, `F` = Frontotemporal Dementia, `C` = Healthy Control).
- `MMSE`: Integer, Mini-Mental State Examination score (0-30).

### EEG Signal Data (`.set` files)
- **Modality**: Resting-state, eyes-closed EEG.
- **Sampling Frequency**: 500 Hz.
- **Units**: Microvolts (µV) post-conversion by MNE-Python (raw dataset uses volts natively).
- **Duration**: Approximately 12 to 13.8 minutes per subject.
- **Number of Channels**: 19
- **Channel Configuration**: 10-20 international system.
- **Channel Names**: 
  - Frontal: Fp1, Fp2, F7, F3, Fz, F4, F8
  - Temporal: T3, T4, T5, T6
  - Central: C3, Cz, C4
  - Parietal: P3, Pz, P4
  - Occipital: O1, O2

### Preprocessed Derivatives Configuration
- Data found under the `derivatives/` directory.
- Re-referenced to A1-A2.
- Band-pass filtered: 0.5 - 45 Hz.
- Cleaned using Artifact Subspace Reconstruction (ASR) and Independent Component Analysis (ICA).

### Model Windows (Epochs)
- **Shape**: `(batch_size, 19, samples)`
- **Duration**: 30 seconds
- **Overlap**: 50% overlap between consecutive windows.
- **Labels**: Each window inherits the diagnostic label of the source participant.
