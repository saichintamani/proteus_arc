import nbformat as nbf

notebook_path = "notebooks/Proteus_Arc_Real_EEG_Experiments_fixed.ipynb"

# Load the notebook
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = nbf.read(f, as_version=4)

# Create the rich markdown cells
feature_extraction_md = nbf.v4.new_markdown_cell("""## Mathematical Formulation of Engineered Features (Candidate 1)

To establish a strong baseline, we extract features across the **Time Domain**, **Frequency Domain**, and **Hjorth Parameters**. 

### 1. Hjorth Parameters
Hjorth parameters describe the statistical properties of the EEG signal $x(t)$ in the time domain:
* **Activity:** Variance of the signal, representing the mean power. $\text{Activity} = \sigma_{x}^2$
* **Mobility:** Mean frequency or proportion of standard deviation of the power spectrum. $\text{Mobility} = \sqrt{\frac{\sigma_{x'}^2}{\sigma_{x}^2}}$
* **Complexity:** Change in frequency, representing how closely the signal resembles a pure sine wave. $\text{Complexity} = \frac{\text{Mobility}(x')}{\text{Mobility}(x)}$

### 2. Frequency Domain (Welch's Method)
We estimate the Power Spectral Density (PSD) using Welch's method to compute power in standard clinical EEG bands:
* **Delta ($\delta$):** 0.5 – 4.0 Hz (Deep sleep / pathological slowing in AD)
* **Theta ($\theta$):** 4.0 – 8.0 Hz (Drowsiness / early dementia marker)
* **Alpha ($\alpha$):** 8.0 – 13.0 Hz (Relaxed wakefulness)
* **Beta ($\beta$):** 13.0 – 30.0 Hz (Active thinking)
""")

baseline_algorithms_md = nbf.v4.new_markdown_cell("""## Traditional Machine Learning Algorithms (Candidate 1)

We evaluate three classical classifiers on the engineered features to form Candidate 1.

1. **Logistic Regression (LR):** A linear classifier that models the probability of a diagnostic class using the logistic function. Excellent for establishing a linear baseline and feature importance.
2. **Support Vector Machine (SVM with RBF Kernel):** Maps features into a high-dimensional space to find a non-linear decision boundary. The RBF kernel allows it to handle complex, non-linear relationships in EEG features.
3. **Random Forest (RF):** An ensemble learning method constructing multiple decision trees during training. It is highly robust to overfitting and handles non-linear feature interactions naturally.
""")

cnn_architecture_md = nbf.v4.new_markdown_cell("""## 1D Convolutional Neural Network (Candidate 2)

Unlike Candidate 1, which relies on hand-crafted features, Candidate 2 learns hierarchical spatial-temporal representations directly from the raw EEG signal.

### Architecture Design
We employ a compact 1D CNN designed specifically to prevent overfitting on small biomedical datasets.

```text
Input EEG Window: (Batch, 19 Channels, 15000 Samples)
       │
       ▼
[ Conv1D ] -> Filters: 16, Kernel: 31, Stride: 1
[ BatchNorm1D ] -> Normalizes activations
[ ReLU ] -> Non-linear activation
[ MaxPool1D ] -> Size: 4, Downsamples temporal dimension
       │
       ▼
[ Conv1D ] -> Filters: 32, Kernel: 15, Stride: 1
[ BatchNorm1D ] -> Normalizes activations
[ ReLU ]
[ MaxPool1D ] -> Size: 4
       │
       ▼
[ AdaptiveAvgPool1D ] -> Pools temporal dimension to exactly 1 (Spatial feature vector)
       │
       ▼
[ Dropout ] -> p=0.5 (Regularization)
[ Linear (Dense) ] -> Output: 3 classes (AD, CN, FTD)
```

**Why this architecture?**
1. **Large initial kernel (31):** Captures broad low-frequency wave morphologies (Delta/Theta).
2. **Adaptive Pooling:** Makes the network invariant to the exact length of the input window.
3. **Low parameter count:** Prevents the model from memorizing the limited number of participants.
""")

# We need to insert these cells into the correct indices.
# Find the indices of specific headers
new_cells = []
for cell in nb.cells:
    if cell.cell_type == 'markdown':
        if "## 11. Baseline Feature Extraction" in cell.source:
            new_cells.append(feature_extraction_md)
        elif "## 12. Train Logistic Regression" in cell.source:
            new_cells.append(baseline_algorithms_md)
        elif "## 15. CNN Initialization" in cell.source:
            new_cells.append(cnn_architecture_md)
    
    new_cells.append(cell)

nb.cells = new_cells

# Save the notebook
with open(notebook_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("Notebook updated successfully with rich algorithms and documentation!")
