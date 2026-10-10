import unittest
import numpy as np
import pandas as pd
from src.proteus_arc.preprocessing.windowing import WindowGenerator

class TestWindowing(unittest.TestCase):
    def setUp(self):
        self.sfreq = 100.0
        self.n_channels = 19
        self.duration_sec = 60.0
        self.n_samples = int(self.sfreq * self.duration_sec)
        self.data = np.random.randn(self.n_channels, self.n_samples)
        self.metadata = {"participant_id": "sub-001", "label": "AD"}
        
    def test_window_generation_shape(self):
        generator = WindowGenerator(epoch_duration_sec=30.0, overlap_ratio=0.5)
        windows, meta_df = generator.generate_windows(self.data, self.sfreq, self.metadata)
        
        expected_window_samples = int(30.0 * 100.0)
        # 60s total, 30s window, 15s step -> starts at 0, 15, 30. Total 3 windows.
        expected_n_windows = 3
        
        self.assertEqual(windows.shape, (expected_n_windows, self.n_channels, expected_window_samples))
        self.assertEqual(len(meta_df), expected_n_windows)
        
    def test_window_metadata(self):
        generator = WindowGenerator(epoch_duration_sec=30.0, overlap_ratio=0.0)
        windows, meta_df = generator.generate_windows(self.data, self.sfreq, self.metadata)
        
        # 60s total, 30s window, 0% overlap -> starts at 0, 30. Total 2 windows.
        self.assertEqual(len(meta_df), 2)
        self.assertEqual(meta_df.iloc[0]["start_sec"], 0.0)
        self.assertEqual(meta_df.iloc[0]["end_sec"], 30.0)
        self.assertEqual(meta_df.iloc[1]["start_sec"], 30.0)
        self.assertEqual(meta_df.iloc[1]["end_sec"], 60.0)
        self.assertEqual(meta_df.iloc[0]["participant_id"], "sub-001")
        
    def test_short_data(self):
        generator = WindowGenerator(epoch_duration_sec=120.0, overlap_ratio=0.5)
        windows, meta_df = generator.generate_windows(self.data, self.sfreq, self.metadata)
        self.assertEqual(windows.shape[0], 0)
        self.assertTrue(meta_df.empty)

if __name__ == "__main__":
    unittest.main()
