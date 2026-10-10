import unittest
import numpy as np
from src.proteus_arc.evaluation.metrics import Evaluator

class TestCNNEvaluation(unittest.TestCase):
    def test_cnn_metrics(self):
        # We reuse the same metric engine, so just asserting basic integration for CNN
        y_true = np.array([0, 1, 2, 0])
        y_pred = np.array([0, 2, 2, 0])
        
        metrics = Evaluator.evaluate_model(y_true, y_pred, classes=[0,1,2])
        self.assertEqual(metrics["Accuracy"], 0.75)
        
        # Test multiclass specific missing metrics logic
        self.assertTrue(np.isnan(metrics["ROC-AUC"]))

if __name__ == "__main__":
    unittest.main()
