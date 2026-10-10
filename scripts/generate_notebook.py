import nbformat as nbf
import os
from pathlib import Path

def create_colab_notebook():
    nb = nbf.v4.new_notebook()
    
    cells = [
        # 1. Environment inspection
        nbf.v4.new_markdown_cell("## 1. Environment Inspection\\nVerify Google Colab GPU and Python versions."),
        nbf.v4.new_code_cell("!nvidia-smi\\n!python --version"),
        
        # 2. Dependency installation
        nbf.v4.new_markdown_cell("## 2. Dependency Installation\\nInstall MNE, Scikit-Learn, PyTorch, and other required scientific libraries."),
        nbf.v4.new_code_cell("!pip install mne scikit-learn pandas scipy pyyaml joblib"),
        
        # 3. Repository access
        nbf.v4.new_markdown_cell("## 3. Repository Access\\nClone the Proteus Arc repository and set up the working directory."),
        nbf.v4.new_code_cell("import os\\n!git clone https://github.com/yourusername/proteus_arc.git\\nos.chdir('proteus_arc')"),
        
        # 4. Dataset access
        nbf.v4.new_markdown_cell("## 4. Dataset Access and Path Configuration\\nConfigure the path to the OpenNeuro ds004504 dataset. You must upload it to Colab, mount Google Drive, or use `openneuro-py`."),
        nbf.v4.new_code_cell("dataset_path = '/content/ds004504' # Update this path to where your dataset is mounted\\n\\nimport os\\nif not os.path.exists(dataset_path):\\n    print(f'ERROR: Dataset not found at {dataset_path}. Please download OpenNeuro ds004504.')\\nelse:\\n    print(f'Dataset found at {dataset_path}.')"),
        
        # 5. Dataset audit
        nbf.v4.new_markdown_cell("## 5. Dataset Audit\\nValidate the metadata using the Proteus Arc data loader."),
        nbf.v4.new_code_cell("from src.proteus_arc.data.loaders import EEGLoader\\n\\nloader = EEGLoader(data_dir=dataset_path, config_path='configs/data.yaml')\\nmeta_report = loader.metadata.validate_metadata()\\nprint(meta_report)"),
        
        # 6. Participant inspection
        nbf.v4.new_markdown_cell("## 6. Participant and Class-Distribution Inspection\\nReview the distribution of AD, FTD, and CN."),
        nbf.v4.new_code_cell("import pandas as pd\\nimport os\\n\\nif meta_report['valid']:\\n    participants_df = pd.read_csv(os.path.join(dataset_path, 'participants.tsv'), sep='\\\\t')\\n    print(participants_df['Group'].value_counts())\\n    print(participants_df.head())"),
        
        # 7. Preprocessing
        nbf.v4.new_markdown_cell("## 7. EEG Preprocessing\\nApply band-pass filters based on `configs/preprocessing.yaml` (0.5-45 Hz)."),
        nbf.v4.new_code_cell("# In a real scenario, loop through the manifest records and apply filtering.\\n# For now, we simulate the loading process safely.\\nfrom src.proteus_arc.preprocessing.filtering import EEGFilter\\nfilter_obj = EEGFilter(l_freq=0.5, h_freq=45.0)"),
        
        # 8. Signal quality
        nbf.v4.new_markdown_cell("## 8. Signal-Quality Checks\\nValidate for flatlines and NaN signals."),
        nbf.v4.new_code_cell("from src.proteus_arc.preprocessing.quality_control import QualityControl\\n# QualityControl.check_nan_inf(data)"),
        
        # 9. Window generation
        nbf.v4.new_markdown_cell("## 9. Window Generation\\nExtract 30-second epochs with 50% overlap."),
        nbf.v4.new_code_cell("from src.proteus_arc.preprocessing.windowing import WindowGenerator\\nwindow_gen = WindowGenerator(epoch_duration_sec=30.0, overlap_ratio=0.5)"),
        
        # 10. Splitting
        nbf.v4.new_markdown_cell("## 10. Participant-Level Train/Validation/Test Splitting\\nStrictly isolate participants into distinct folds (60/20/20)."),
        nbf.v4.new_code_cell("from src.proteus_arc.evaluation.participant_split import ParticipantSplitter\\n# Ensure X, y, groups are fully loaded here.\\n# splitter_test = ParticipantSplitter(method='group_shuffle', test_size=0.2)"),
        
        # 11. Feature Extraction
        nbf.v4.new_markdown_cell("## 11. Baseline Feature Extraction\\nExtract Time, Frequency, and Hjorth parameters for baseline ML."),
        nbf.v4.new_code_cell("from src.proteus_arc.features.feature_pipeline import FeaturePipeline\\npipeline = FeaturePipeline(config_path='configs/baseline.yaml', sfreq=500.0)"),
        
        # 12-14. Baseline Training
        nbf.v4.new_markdown_cell("## 12-14. Train Candidate 1: Baseline Models (LR, SVM, RF)"),
        nbf.v4.new_code_cell("from src.proteus_arc.models.baseline_models import BaselineModels\\n# models = BaselineModels().get_all_models()\\n# Iterate and fit models using X_tr_features, y_tr..."),
        
        # 15. CNN Init
        nbf.v4.new_markdown_cell("## 15. CNN Initialization (Candidate 2)\\nInitialize the 1D CNN architecture."),
        nbf.v4.new_code_cell("from src.proteus_arc.models.cnn_1d import EEG1DCNN\\nmodel = EEG1DCNN(in_channels=19, num_classes=3)"),
        
        # 16. CNN Training
        nbf.v4.new_markdown_cell("## 16. CNN Training and Validation"),
        nbf.v4.new_code_cell("from src.proteus_arc.models.training import CNNTrainer\\n# trainer = CNNTrainer(model, learning_rate=0.001)\\n# history = trainer.fit(...)"),
        
        # 17-18. Final Test Eval & Participant Aggregation
        nbf.v4.new_markdown_cell("## 17-18. Final Held-Out Test Evaluation & Participant Aggregation\\nEvaluate models on the held-out test partition using subject-level majority voting."),
        nbf.v4.new_code_cell("from src.proteus_arc.evaluation.metrics import Evaluator\\n# Epoch-level and Subject-level evaluation logic goes here."),
        
        # 19. Metrics and Confusion
        nbf.v4.new_markdown_cell("## 19. Metrics and Confusion Matrices"),
        nbf.v4.new_code_cell("# Evaluator.evaluate_model(...) returns confusion matrix and metrics dict."),
        
        # 20. Comparison
        nbf.v4.new_markdown_cell("## 20. Baseline vs. CNN Comparison\\nTabulate the accuracies of LR, SVM, RF, and CNN."),
        nbf.v4.new_code_cell("# Build a pandas DataFrame to display results cleanly."),
        
        # 21. Error Analysis
        nbf.v4.new_markdown_cell("## 21. Error Analysis\\nInvestigate specific participants that all models fail on."),
        nbf.v4.new_code_cell("# Save misclassifications to a DataFrame for review."),
        
        # 22. Saving
        nbf.v4.new_markdown_cell("## 22. Saving Experiment Artifacts\\nExport models and metrics to Google Drive."),
        nbf.v4.new_code_cell("# joblib.dump, torch.save...\\n# !zip -r outputs.zip outputs/")
    ]
    
    nb['cells'] = cells
    
    Path('notebooks').mkdir(exist_ok=True)
    with open('notebooks/Proteus_Arc_Real_EEG_Experiments.ipynb', 'w') as f:
        nbf.write(nb, f)
        
if __name__ == '__main__':
    create_colab_notebook()
