import unittest
import numpy as np
from src.proteus_arc.evaluation.metrics import Evaluator

class TestMetrics(unittest.TestCase):
    def test_evaluate_model_binary(self):
        y_true = np.array(["AD", "AD", "CN", "CN"])
        y_pred = np.array(["AD", "CN", "CN", "CN"])
        y_proba = np.array([[0.8, 0.2], [0.4, 0.6], [0.1, 0.9], [0.2, 0.8]])
        
        metrics = Evaluator.evaluate_model(y_true, y_pred, y_proba, classes=["AD", "CN"])
        self.assertEqual(metrics["Accuracy"], 0.75)
        self.assertIn("Confusion Matrix", metrics)
        self.assertIn("Classification Report", metrics)
        self.assertTrue(np.isfinite(metrics["ROC-AUC"]))
        
    def test_subject_level_vote(self):
        y_true = np.array(["AD", "AD", "AD", "CN", "CN", "CN"])
        y_pred = np.array(["AD", "CN", "AD", "CN", "AD", "CN"])
        groups = np.array(["s1", "s1", "s1", "s2", "s2", "s2"])
        
        t, p = Evaluator.subject_level_vote(y_true, y_pred, groups)
        self.assertEqual(list(t), ["AD", "CN"])
        self.assertEqual(list(p), ["AD", "CN"])

if __name__ == "__main__":
    unittest.main()
