import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix, roc_auc_score
)
from typing import Dict, Any, Tuple, List

class Evaluator:
    @staticmethod
    def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray = None, classes: List = None) -> Dict[str, Any]:
        """
        Calculates all required metrics for the baseline evaluations.
        """
        metrics = {}
        metrics["Accuracy"] = accuracy_score(y_true, y_pred)
        metrics["Balanced Accuracy"] = balanced_accuracy_score(y_true, y_pred)
        metrics["Macro Precision"] = precision_score(y_true, y_pred, average="macro", zero_division=0)
        metrics["Macro Recall"] = recall_score(y_true, y_pred, average="macro", zero_division=0)
        metrics["Macro F1"] = f1_score(y_true, y_pred, average="macro", zero_division=0)
        
        # Calculate sensitivity/specificity for binary tasks (or OVR logic)
        unique_classes = np.unique(y_true)
        if len(unique_classes) == 2:
            # Assume 2nd class is positive class if not explicitly defined
            pos_class = unique_classes[1]
            neg_class = unique_classes[0]
            
            tp = np.sum((y_true == pos_class) & (y_pred == pos_class))
            tn = np.sum((y_true == neg_class) & (y_pred == neg_class))
            fn = np.sum((y_true == pos_class) & (y_pred == neg_class))
            fp = np.sum((y_true == neg_class) & (y_pred == pos_class))
            
            metrics["Sensitivity"] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            metrics["Specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            
            if y_proba is not None:
                # y_proba should be shape (n, 2)
                if y_proba.ndim == 2 and y_proba.shape[1] == 2:
                    metrics["ROC-AUC"] = roc_auc_score(y_true == pos_class, y_proba[:, 1])
                else:
                    metrics["ROC-AUC"] = np.nan
        else:
            metrics["Sensitivity"] = np.nan # Undefined for multiclass simply, handle per-class in report
            metrics["Specificity"] = np.nan
            if y_proba is not None and y_proba.ndim == 2:
                try:
                    metrics["ROC-AUC"] = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
                except ValueError:
                    metrics["ROC-AUC"] = np.nan
            else:
                metrics["ROC-AUC"] = np.nan
                
        metrics["Confusion Matrix"] = confusion_matrix(y_true, y_pred, labels=classes).tolist()
        metrics["Classification Report"] = classification_report(y_true, y_pred, labels=classes, zero_division=0, output_dict=True)
        
        return metrics

    @staticmethod
    def subject_level_vote(y_true: np.ndarray, y_pred: np.ndarray, groups: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculates the subject-level majority vote.
        """
        df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "group": groups})
        
        grouped_true = df.groupby("group")["y_true"].first()
        grouped_pred = df.groupby("group")["y_pred"].agg(lambda x: x.value_counts().idxmax())
        
        return grouped_true.values, grouped_pred.values
