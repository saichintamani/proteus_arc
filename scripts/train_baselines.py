import os
import argparse
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import cross_val_predict

from src.proteus_arc.data.loaders import EEGLoader
from src.proteus_arc.preprocessing.windowing import WindowGenerator
from src.proteus_arc.features.feature_pipeline import FeaturePipeline
from src.proteus_arc.evaluation.participant_split import ParticipantSplitter
from src.proteus_arc.models.baseline_models import BaselineModels
from src.proteus_arc.evaluation.metrics import Evaluator

def main():
    print("=" * 60)
    print("PROTEUS ARC: BASELINE ML PIPELINE (CANDIDATE 1)")
    print("=" * 60)
    
    # Configuration
    data_dir = "ds004504"
    config_dir = "configs"
    
    loader = EEGLoader(data_dir=data_dir, config_path=os.path.join(config_dir, "data.yaml"))
    meta_report = loader.metadata.validate_metadata()
    
    if not meta_report["valid"]:
        print(f"Dataset missing or invalid at {data_dir}. Errors: {meta_report['errors']}")
        print("Falling back to small SYNTHETIC software fixture for code validation.")
        
        # Synthetic Data Generation
        n_participants = 10
        n_epochs = 5
        n_channels = 19
        sfreq = 500.0
        n_samples = int(sfreq * 30.0) # 30 sec
        
        X = np.random.randn(n_participants * n_epochs, n_channels, n_samples)
        y = np.array(["AD", "CN"]).repeat(n_participants // 2 * n_epochs)
        groups = np.array([f"sub-{i}" for i in range(n_participants)]).repeat(n_epochs)
        channel_names = loader.config.get("channels", [f"ch_{i}" for i in range(n_channels)])
    else:
        print("Found real dataset metadata. Extracting windows...")
        manifest = loader.discover_recordings(check_mne=True)
        # Note: In a full pipeline, we would loop through manifest records, load MNE files, 
        # apply filtering, epoching, and concatenate into X. 
        # For memory safety in this demonstration phase without a cluster, we assert limitations.
        raise NotImplementedError("Real data epoch loading is staged but not fully integrated to avoid OOM errors locally.")
        
    print(f"Data shape: {X.shape}, Classes: {np.unique(y)}, Participants: {len(np.unique(groups))}")
    
    # Feature Extraction
    print("Extracting features...")
    pipeline = FeaturePipeline(config_path=os.path.join(config_dir, "baseline.yaml"), sfreq=sfreq, channel_names=channel_names)
    X_features_df = pipeline.transform(X)
    X_features = X_features_df.to_numpy(dtype=float)
    
    # Participant Split
    print("Applying participant-level split...")
    splitter = ParticipantSplitter(method="group_shuffle", test_size=0.2)
    X_tr, X_te, y_tr, y_te, g_tr, g_te = splitter.split(X_features, y, groups)
    
    # Model Training and Evaluation
    models = BaselineModels(config_path=os.path.join(config_dir, "baseline.yaml")).get_all_models()
    classes = list(np.unique(y))
    
    output_dir = Path("outputs/metrics")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for name, model in models.items():
        print(f"\\nEvaluating {name}...")
        # Train
        model.fit(X_tr, y_tr)
        
        # Test
        y_pred = model.predict(X_te)
        y_proba = model.predict_proba(X_te) if hasattr(model, "predict_proba") else None
        
        # Evaluate Epoch Level
        metrics = Evaluator.evaluate_model(y_te, y_pred, y_proba, classes=classes)
        print(f"Epoch-Level Acc: {metrics['Accuracy']:.4f} | F1: {metrics['Macro F1']:.4f}")
        
        # Evaluate Subject Level
        y_true_sub, y_pred_sub = Evaluator.subject_level_vote(y_te, y_pred, g_te)
        sub_metrics = Evaluator.evaluate_model(y_true_sub, y_pred_sub, classes=classes)
        print(f"Subject-Level Acc: {sub_metrics['Accuracy']:.4f}")
        
        # Save model
        chk_dir = Path("outputs/checkpoints")
        chk_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, chk_dir / f"{name.replace(' ', '_').lower()}.joblib")

    print("\\nPipeline execution complete. Models saved to outputs/checkpoints.")

if __name__ == "__main__":
    main()
