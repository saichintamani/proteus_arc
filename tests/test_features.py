import unittest
import numpy as np
from src.proteus_arc.features.time_features import TimeDomainFeatures
from src.proteus_arc.features.frequency_features import FrequencyDomainFeatures
from src.proteus_arc.features.feature_pipeline import FeaturePipeline

class TestFeatures(unittest.TestCase):
    def setUp(self):
        self.signal = np.random.randn(1000)
        self.channel = "Fp1"
        self.sfreq = 500.0
        
    def test_time_features(self):
        features = TimeDomainFeatures.extract(self.signal, self.channel)
        self.assertIn(f"{self.channel}_mean", features)
        self.assertIn(f"{self.channel}_hjorth_complexity", features)
        # Should not have NaN
        for v in features.values():
            self.assertTrue(np.isfinite(v))
            
    def test_frequency_features(self):
        bands = {"alpha": [8.0, 13.0]}
        extractor = FrequencyDomainFeatures(bands, self.sfreq)
        features = extractor.extract(self.signal, self.channel)
        self.assertIn(f"{self.channel}_alpha_power", features)
        self.assertIn(f"{self.channel}_alpha_relative_power", features)
        
    def test_pipeline_shapes(self):
        # Need to create dummy baseline.yaml first since FeaturePipeline loads it
        import yaml
        import tempfile
        import os
        from pathlib import Path
        
        temp_dir = tempfile.TemporaryDirectory()
        config_path = Path(temp_dir.name) / "baseline.yaml"
        config = {
            "features": {
                "frequency_domain": {
                    "bands": {"delta": [0.5, 4.0]}
                }
            }
        }
        with open(config_path, "w") as f:
            yaml.dump(config, f)
            
        pipeline = FeaturePipeline(config_path=str(config_path), sfreq=256.0, channel_names=["C3", "C4"])
        windows = np.random.randn(2, 2, 256*10) # 2 windows, 2 channels, 10s
        df = pipeline.transform(windows)
        
        # 2 windows -> 2 rows
        self.assertEqual(len(df), 2)
        # 2 channels * (9 time + 3 hjorth + 2 freq) = 2 * 14 = 28 features
        self.assertEqual(len(df.columns), 28)
        
        temp_dir.cleanup()

if __name__ == "__main__":
    unittest.main()
