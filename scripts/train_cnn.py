import os
import argparse
import yaml
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder

from src.proteus_arc.data.loaders import EEGLoader
from src.proteus_arc.evaluation.participant_split import ParticipantSplitter
from src.proteus_arc.models.cnn_1d import EEG1DCNN
from src.proteus_arc.models.training import EEGDataset, CNNTrainer
from src.proteus_arc.evaluation.metrics import Evaluator

def main():
    print("=" * 60)
    print("PROTEUS ARC: 1D CNN PIPELINE (CANDIDATE 2)")
    print("=" * 60)
    
    config_dir = "configs"
    data_dir = "ds004504"
    
    with open(os.path.join(config_dir, "cnn.yaml"), "r") as f:
        cnn_config = yaml.safe_load(f)
        
    loader = EEGLoader(data_dir=data_dir, config_path=os.path.join(config_dir, "data.yaml"))
    meta_report = loader.metadata.validate_metadata()
    
    if not meta_report["valid"]:
        print(f"Dataset missing or invalid at {data_dir}.")
        print("Falling back to small SYNTHETIC software fixture for code validation.")
        
        n_participants = 15
        n_epochs = 5
        n_channels = cnn_config["cnn"]["architecture"]["in_channels"]
        sfreq = 500.0
        n_samples = int(sfreq * 30.0)
        
        X = np.random.randn(n_participants * n_epochs, n_channels, n_samples)
        y = np.array(["AD", "CN", "FTD"]).repeat(5 * n_epochs)
        groups = np.array([f"sub-{i}" for i in range(n_participants)]).repeat(n_epochs)
    else:
        print("Found real dataset metadata. Loading epoch tensors...")
        raise NotImplementedError("Real data tensor generation is pending until dataset is mounted.")

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Participant Split (Train, Val, Test)
    # 1. Split Train+Val / Test
    print("Applying participant-level isolation splits...")
    splitter_test = ParticipantSplitter(method="group_shuffle", test_size=0.2, random_state=42)
    X_temp, X_te, y_temp, y_te, g_temp, g_te = splitter_test.split(X, y_encoded, groups)
    
    # 2. Split Train / Val
    splitter_val = ParticipantSplitter(method="group_shuffle", test_size=0.25, random_state=42) # 0.25 of 0.8 = 0.2 overall
    X_tr, X_va, y_tr, y_va, g_tr, g_va = splitter_val.split(X_temp, y_temp, g_temp)
    
    print(f"Train participants: {len(np.unique(g_tr))} | Val participants: {len(np.unique(g_va))} | Test participants: {len(np.unique(g_te))}")
    
    # Dataloaders
    batch_size = cnn_config["cnn"]["training"]["batch_size"]
    train_dataset = EEGDataset(X_tr, y_tr, g_tr)
    val_dataset = EEGDataset(X_va, y_va, g_va)
    test_dataset = EEGDataset(X_te, y_te, g_te)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Model
    model = EEG1DCNN(**cnn_config["cnn"]["architecture"])
    
    # Class weights for loss
    class_counts = np.bincount(y_tr)
    total = len(y_tr)
    class_weights = torch.tensor([total / c for c in class_counts], dtype=torch.float32)
    
    trainer = CNNTrainer(
        model=model, 
        learning_rate=cnn_config["cnn"]["training"]["learning_rate"],
        weight_decay=cnn_config["cnn"]["training"]["weight_decay"],
        patience=cnn_config["cnn"]["training"]["early_stopping_patience"],
        class_weights=class_weights
    )
    
    # Train
    print("\\nStarting training...")
    save_path = "outputs/checkpoints/best_cnn.pth"
    history = trainer.fit(train_loader, val_loader, epochs=cnn_config["cnn"]["training"]["epochs"], save_path=save_path)
    
    # Final Test Evaluation (on best model)
    print("\\nEvaluating Best Checkpoint on Hold-Out Test Set...")
    trainer.model.load_state_dict(torch.load(save_path, weights_only=True))
    _, test_acc, test_f1, all_labels, all_preds, all_groups = trainer.val_epoch(test_loader)
    
    classes = list(range(len(le.classes_)))
    
    # Epoch Level
    print(f"Epoch-Level Acc: {test_acc:.4f} | F1: {test_f1:.4f}")
    
    # Subject Level
    y_true_sub, y_pred_sub = Evaluator.subject_level_vote(all_labels, all_preds, all_groups)
    sub_metrics = Evaluator.evaluate_model(y_true_sub, y_pred_sub, classes=classes)
    print(f"Subject-Level Acc: {sub_metrics['Accuracy']:.4f}")
    
    print("\\nCNN Pipeline execution complete.")

if __name__ == "__main__":
    main()
