import unittest
from src.proteus_arc.models.baseline_models import BaselineModels
import numpy as np

class TestModels(unittest.TestCase):
    def test_pipeline_construction(self):
        # Create dummy config
        import yaml
        import tempfile
        from pathlib import Path
        
        temp_dir = tempfile.TemporaryDirectory()
        config_path = Path(temp_dir.name) / "baseline.yaml"
        with open(config_path, "w") as f:
            yaml.dump({"models": {"logistic_regression": {"max_iter": 100}, "svm": {}, "random_forest": {}}}, f)
            
        bm = BaselineModels(str(config_path))
        models = bm.get_all_models()
        self.assertIn("Logistic Regression", models)
        self.assertIn("SVM", models)
        self.assertIn("Random Forest", models)
        
        # Test basic fitting on synthetic array
        X = np.random.randn(10, 5)
        y = np.array([0, 1] * 5)
        
        for name, model in models.items():
            model.fit(X, y)
            pred = model.predict(X)
            self.assertEqual(pred.shape, (10,))
            
        temp_dir.cleanup()

if __name__ == "__main__":
    unittest.main()
