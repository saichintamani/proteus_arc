import numpy as np
from typing import List, Tuple
import pandas as pd

class WindowGenerator:
    """
    Generates consistent windows from continuous EEG recordings.
    """
    def __init__(self, epoch_duration_sec: float = 30.0, overlap_ratio: float = 0.5):
        self.epoch_duration_sec = epoch_duration_sec
        self.overlap_ratio = overlap_ratio
        
    def generate_windows(self, data: np.ndarray, sfreq: float, metadata: dict) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Segment data (channels, samples) into windows.
        Returns:
            windows: np.ndarray of shape (n_windows, channels, window_samples)
            window_metadata: pd.DataFrame with metadata per window
        """
        if data.ndim != 2:
            raise ValueError(f"Data must be 2D (channels, samples). Got {data.ndim}D.")
            
        n_channels, n_samples = data.shape
        window_samples = int(self.epoch_duration_sec * sfreq)
        
        if window_samples <= 0:
            raise ValueError("Window length must be > 0.")
            
        if window_samples > n_samples:
            # Signal is shorter than window duration
            return np.empty((0, n_channels, window_samples)), pd.DataFrame()
            
        step_samples = max(1, int(window_samples * (1 - self.overlap_ratio)))
        
        starts = range(0, n_samples - window_samples + 1, step_samples)
        windows = np.stack([data[:, s:s + window_samples] for s in starts])
        
        # Build metadata for each window
        base_meta = metadata.copy()
        
        meta_list = []
        for i, s in enumerate(starts):
            meta = base_meta.copy()
            meta["window_index"] = i
            meta["start_sec"] = s / sfreq
            meta["end_sec"] = (s + window_samples) / sfreq
            meta["sfreq"] = sfreq
            meta["n_channels"] = n_channels
            meta_list.append(meta)
            
        return windows, pd.DataFrame(meta_list)
