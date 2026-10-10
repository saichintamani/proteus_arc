# Dataset Setup Guide — OpenNeuro ds004504

## Dataset Identity
- **Name:** A dataset of EEG recordings from: Alzheimer's disease, Frontotemporal dementia and Healthy subjects
- **Identifier:** OpenNeuro ds004504
- **Authors:** Miltiadous, A. et al.
- **Publication:** Data 8(6):95, 2023
- **DOI:** [10.3390/data8060095](https://doi.org/10.3390/data8060095)
- **License:** CC0 (Public Domain)
- **URL:** https://openneuro.org/datasets/ds004504

## Verified Dataset Characteristics

### Participants
| Group | Code | Count | Description |
|-------|------|-------|-------------|
| AD    | A    | 36    | Alzheimer's Disease |
| FTD   | F    | 23    | Frontotemporal Dementia |
| CN    | C    | 29    | Healthy Controls |
| **Total** | | **88** | |

### Metadata Columns (`participants.tsv`)
| Column | Type | Description |
|--------|------|-------------|
| `participant_id` | String | `sub-001` through `sub-088` |
| `Gender` | Categorical | `M` or `F` |
| `Age` | Integer | Participant age at recording |
| `Group` | Categorical | `A`, `C`, or `F` |
| `MMSE` | Integer | Mini-Mental State Examination score |

### EEG Recording Configuration
- **Paradigm:** Resting state, eyes closed, sitting position
- **Sampling frequency:** 500 Hz
- **Electrode system:** International 10-20 system, 19 scalp electrodes
- **Channels:** Fp1, Fp2, F7, F3, Fz, F4, F8, T3, C3, Cz, C4, T4, T5, P3, Pz, P4, T6, O1, O2
- **Reference:** Re-referenced to average of A1-A2 (earlobes)
- **File format:** EEGLAB `.set` files (in `derivatives/`)

### Preprocessing (Applied by Dataset Authors)
1. Band-pass filter: 0.5–45 Hz (Butterworth)
2. Re-reference to A1-A2
3. Artifact Subspace Reconstruction (ASR)
4. Independent Component Analysis (ICA) with ICLabel

### Known Limitations
- **Single-site study** — generalization to other clinical settings is unverified.
- **Class imbalance** — FTD group (23) is smaller than AD (36) and CN (29).
- **CN MMSE scores** — All healthy controls have MMSE = 30 (ceiling effect), making the task easier than a general clinical screening scenario.
- **Pre-split preprocessing** — ASR and ICA were applied to full continuous recordings before any train/test partitioning. This is a minor potential source of information leakage that is inherent to this specific dataset and cannot be retroactively corrected without access to the raw recordings.

## Download Procedure

### Option 1: `openneuro-py` (Recommended for Colab)
```bash
pip install openneuro-py
openneuro-py download --dataset ds004504 \
    --target_dir /content/ds004504 \
    --include 'participants.tsv' 'participants.json' 'dataset_description.json' 'derivatives/*'
```

### Option 2: Manual Download
1. Visit https://openneuro.org/datasets/ds004504
2. Click **Download** and select the derivatives.
3. Extract to a known directory.
4. Update `DATASET_PATH` in the notebook.

### Option 3: AWS S3 (Advanced)
```bash
aws s3 sync --no-sign-request s3://openneuro.org/ds004504 /path/to/ds004504
```

## Expected Directory Structure
```
ds004504/
├── dataset_description.json
├── participants.tsv
├── participants.json
└── derivatives/
    ├── sub-001/
    │   └── eeg/
    │       └── sub-001_task-eyesclosed_eeg.set
    │       └── sub-001_task-eyesclosed_eeg.fdt
    ├── sub-002/
    │   └── eeg/
    │       └── ...
    └── ...
```

## Configuration Mapping
The project configuration (`configs/data.yaml`) maps dataset labels:
- `A` → `AD` (Alzheimer's Disease)
- `C` → `CN` (Cognitively Normal / Healthy Control)
- `F` → `FTD` (Frontotemporal Dementia)

These have been verified against the actual `participants.tsv` downloaded from GitHub.
