import unittest
from pathlib import Path
from src.proteus_arc.data.audit import audit_dataset

class TestDatasetAudit(unittest.TestCase):
    def test_missing_dataset(self):
        # Pointing to a non-existent directory
        report = audit_dataset("non_existent_data_dir_12345")
        self.assertFalse(report["dataset_exists"])
        self.assertFalse(report["participants_file_exists"])
        self.assertIsNotNone(report["missing_files_error"])
        self.assertIn("not found", report["missing_files_error"])

if __name__ == "__main__":
    unittest.main()
