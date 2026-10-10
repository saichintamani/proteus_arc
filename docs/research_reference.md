# Research References

## 1. Primary Dataset Descriptor
- **Title**: A dataset of EEG recordings from: Alzheimer's disease, Frontotemporal dementia and Healthy subjects
- **Authors**: Miltiadous et al.
- **Publication Year**: 2023
- **Venue**: Data 8(6):95
- **DOI**: 10.3390/data8060095
- **Relevance to Proteus Arc**: This is the primary dataset (`ds004504`) used in the experiments. It provides the preprocessed and raw EEG data, demographic labels, and resting-state (eyes closed) paradigm.

## 2. Baseline & Proposed Graph Approach Context
- **Title**: DICE-Net: A Novel Convolution-Transformer Architecture for Alzheimer Detection in EEG Signals
- **Authors**: Miltiadous et al.
- **Publication Year**: 2023
- **Venue**: IEEE Access 11:71840
- **DOI**: 10.1109/ACCESS.2023.3294618
- **Relevance to Proteus Arc**: Mentions an 83.28% AD vs CN accuracy benchmark using Leave-One-Subject-Out validation, acting as the state-of-the-art benchmark context.

## Extracted Experimental Protocols
1. **Research objective**: Alzheimer's Disease Classification from resting-state EEG (AD vs. CN, FTD vs. CN, AD vs. FTD, 3-class).
2. **Dataset source**: OpenNeuro ds004504
3. **Participant population and class definitions**: 88 subjects total (36 AD, 23 FTD, 29 CN).
4. **Number of independent participants**: 88
5. **EEG recording configuration**: Resting state, eyes closed, sitting position.
6. **Sampling frequency and electrode configuration**: 500 Hz, 19 scalp electrodes (10-20 system: Fp1, Fp2, F7, F3, Fz, F4, F8, T3, C3, Cz, C4, T4, T5, P3, Pz, P4, T6, O1, O2).
7. **Preprocessing methodology**: Butterworth band-pass 0.5-45 Hz, re-reference to A1-A2, Artifact Subspace Reconstruction (ASR), ICA with ICLabel.
8. **Windowing strategy**: 30s epochs with 50% overlap.
9. **Model architecture**: 
   - Original paper benchmarks: SVM, RF, DT.
   - Proteus Arc Candidates: Traditional ML (Candidate 1), 1D CNN (Candidate 2), and Graph/T-GCN baseline (future).
10. **Training and evaluation procedure**: Participant-independent validation (StratifiedGroupKFold or Leave-One-Subject-Out).
11. **Participant split methodology**: Splitting by participant ID (subject-wise grouping).
12. **Reported metrics**: AD vs CN: SVM+PCA ~73.75%, RF ~77% (from prior literature on this dataset), DICE-Net 83.28%.
13. **Limitations and possible sources of leakage**: High chance of epoch leakage if split randomly due to correlation between windows from the same subject. The dataset is small (88 subjects) with class imbalance, single-site, and healthy controls have perfect MMSE scores (30). Preprocessing (ASR/ICA) was applied across the full recordings prior to splits.
