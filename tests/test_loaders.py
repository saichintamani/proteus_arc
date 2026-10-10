import unittest
from pathlib import Path
import tempfile
import os
import pandas as pd
import yaml
from src.proteus_arc.data.metadata import DatasetMetadata
from src.proteus_arc.data.manifest import DatasetManifest
from src.proteus_arc.data.loaders import EEGLoader

class TestDatasetIngestion(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.test_dir.name) / "ds004504"
        self.data_dir.mkdir()
        self.config_path = Path(self.test_dir.name) / "data.yaml"
        
        # Create config
        config = {
            "dataset": {
                "classes": ["A", "C"],
                "mapping": {"A": "AD", "C": "CN"}
            }
        }
        with open(self.config_path, "w") as f:
            yaml.dump(config, f)
            
    def tearDown(self):
        self.test_dir.cleanup()
        
    def test_missing_metadata(self):
        loader = EEGLoader(str(self.data_dir), str(self.config_path))
        report = loader.metadata.validate_metadata()
        self.assertFalse(report["valid"])
        self.assertIn("participants.tsv is missing or empty.", report["errors"])
        
    def test_valid_metadata_and_discovery(self):
        # Create dummy participants.tsv
        df = pd.DataFrame({
            "participant_id": ["sub-001", "sub-002"],
            "Group": ["A", "C"],
            "Age": [70, 65]
        })
        df.to_csv(self.data_dir / "participants.tsv", sep='\t', index=False)
        
        # Create dummy derivative files
        for sub in ["sub-001", "sub-002"]:
            eeg_dir = self.data_dir / "derivatives" / sub / "eeg"
            eeg_dir.mkdir(parents=True)
            (eeg_dir / f"{sub}_task-eyesclosed_eeg.set").touch()
            
        loader = EEGLoader(str(self.data_dir), str(self.config_path))
        report = loader.metadata.validate_metadata()
        self.assertTrue(report["valid"])
        self.assertEqual(report["participants_count"], 2)
        
        # Discover recordings (without MNE check to avoid needing real .set file formats)
        manifest = loader.discover_recordings(check_mne=False)
        summary = manifest.get_summary()
        self.assertEqual(summary["total_recordings"], 2)
        self.assertEqual(summary["class_distribution"], {"AD": 1, "CN": 1})
        
    def test_manifest_save_load(self):
        manifest = DatasetManifest()
        manifest.add_recording("sub-001", "AD", "path/to/file.set", 500.0, ["Fp1", "Fp2"], 300.0)
        
        save_path = Path(self.test_dir.name) / "manifest.csv"
        manifest.save(str(save_path))
        
        loaded_manifest = DatasetManifest.load(str(save_path))
        summary = loaded_manifest.get_summary()
        self.assertEqual(summary["total_recordings"], 1)
        self.assertEqual(summary["participants"], 1)

if __name__ == "__main__":
    unittest.main()
