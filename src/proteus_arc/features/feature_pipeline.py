import pandas as pd
import numpy as np
from typing import List, Dict
import yaml
from src.proteus_arc.features.time_features import TimeDomainFeatures
from src.proteus_arc.features.frequency_features import FrequencyDomainFeatures

class FeaturePipeline:
    def __init__(self, config_path: str = "configs/baseline.yaml", sfreq: float = 500.0, channel_names: List[str] = None):
        self.sfreq = sfreq
        self.channel_names = channel_names if channel_names else []
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)["features"]
            
        self.freq_extractor = FrequencyDomainFeatures(
            bands=self.config["frequency_domain"]["bands"],
            sfreq=self.sfreq
        )

    def extract_epoch(self, epoch: np.ndarray) -> Dict[str, float]:
        """
        Extract features for a single epoch (channels, samples)
        """
        if epoch.ndim != 2:
            raise ValueError("Epoch must be (channels, samples)")
            
        if not self.channel_names:
            self.channel_names = [f"ch_{i}" for i in range(epoch.shape[0])]
            
        if len(self.channel_names) != epoch.shape[0]:
            raise ValueError("Mismatch between channel names and channel dimension.")
            
        epoch_features = {}
        for ch_idx, ch_name in enumerate(self.channel_names):
            signal = epoch[ch_idx]
            # Time domain
            t_feats = TimeDomainFeatures.extract(signal, ch_name)
            epoch_features.update(t_feats)
            # Frequency domain
            f_feats = self.freq_extractor.extract(signal, ch_name)
            epoch_features.update(f_feats)
            
        return epoch_features
        
    def transform(self, windows: np.ndarray) -> pd.DataFrame:
        """
        Transforms multiple windows (n_windows, channels, samples) into a DataFrame.
        """
        rows = []
        for i in range(windows.shape[0]):
            rows.append(self.extract_epoch(windows[i]))
        return pd.DataFrame(rows)
