import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import time
import copy
from typing import Dict, Any, Tuple
from src.proteus_arc.evaluation.metrics import Evaluator

class EEGDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray):
        """
        X: (N, channels, time)
        y: (N,)
        groups: (N,)
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        # Assuming y is categorical/string, we need it as Long tensor for CrossEntropyLoss
        # Mapping needs to be handled before passing here, or we do it during creation.
        # We assume y is already integer-encoded here.
        self.y = torch.tensor(y, dtype=torch.long)
        self.groups = groups
        
    def __len__(self):
        return len(self.y)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.groups[idx]

class CNNTrainer:
    def __init__(self, model: nn.Module, device: str = None, 
                 learning_rate: float = 0.001, weight_decay: float = 1e-4, 
                 patience: int = 10, class_weights: torch.Tensor = None):
        
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = model.to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.device) if class_weights is not None else None)
        self.patience = patience
        
    def train_epoch(self, dataloader: DataLoader) -> Tuple[float, float, float]:
        self.model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        for X_batch, y_batch, _ in dataloader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(X_batch)
            loss = self.criterion(outputs, y_batch)
            
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item() * X_batch.size(0)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
            
        epoch_loss = running_loss / len(dataloader.dataset)
        metrics = Evaluator.evaluate_model(np.array(all_labels), np.array(all_preds))
        return epoch_loss, metrics.get("Accuracy", 0.0), metrics.get("Macro F1", 0.0)
        
    def val_epoch(self, dataloader: DataLoader) -> Tuple[float, float, float, np.ndarray, np.ndarray, np.ndarray]:
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        all_groups = []
        
        with torch.no_grad():
            for X_batch, y_batch, g_batch in dataloader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                
                running_loss += loss.item() * X_batch.size(0)
                preds = torch.argmax(outputs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())
                all_groups.extend(g_batch)
                
        epoch_loss = running_loss / len(dataloader.dataset)
        metrics = Evaluator.evaluate_model(np.array(all_labels), np.array(all_preds))
        return epoch_loss, metrics.get("Accuracy", 0.0), metrics.get("Macro F1", 0.0), np.array(all_labels), np.array(all_preds), np.array(all_groups)
        
    def fit(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int, save_path: str) -> Dict[str, list]:
        history = {"train_loss": [], "val_loss": [], "val_f1": []}
        best_val_loss = float("inf")
        best_model_wts = copy.deepcopy(self.model.state_dict())
        patience_counter = 0
        best_epoch = 0
        
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        
        for epoch in range(epochs):
            start_time = time.time()
            
            train_loss, train_acc, train_f1 = self.train_epoch(train_loader)
            val_loss, val_acc, val_f1, _, _, _ = self.val_epoch(val_loader)
            
            duration = time.time() - start_time
            
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_f1"].append(val_f1)
            
            print(f"Epoch {epoch+1}/{epochs} [{duration:.1f}s] - "
                  f"Train Loss: {train_loss:.4f}, F1: {train_f1:.4f} | "
                  f"Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}")
                  
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                best_model_wts = copy.deepcopy(self.model.state_dict())
                torch.save(best_model_wts, save_path)
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= self.patience:
                print(f"Early stopping triggered. Best epoch was {best_epoch}.")
                break
                
        self.model.load_state_dict(best_model_wts)
        return history
