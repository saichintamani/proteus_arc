import numpy as np
from scipy.signal import welch
from typing import Dict

class FrequencyDomainFeatures:
    def __init__(self, bands: Dict[str, list], sfreq: float):
        self.bands = bands
        self.sfreq = sfreq
        
        # NumPy >= 2.0 has np.trapezoid, fallback to np.trapz
        self.trapz = getattr(np, "trapezoid", None) or np.trapz

    def extract(self, signal: np.ndarray, channel_name: str) -> Dict[str, float]:
        """
        Extract frequency-domain band power features using Welch's method.
        """
        if signal.ndim != 1:
            raise ValueError("Signal must be 1D.")
            
        n_samples = len(signal)
        if n_samples < 8:
            raise ValueError("Signal too short for PSD calculation.")
            
        freqs, psd = welch(
            signal, 
            fs=self.sfreq, 
            nperseg=min(n_samples, max(8, int(self.sfreq * 2)))
        )
        
        features = {}
        prefix = str(channel_name)
        
        total_mask = (freqs >= 0.5) & (freqs <= min(45.0, self.sfreq / 2.0))
        total_power = (
            self.trapz(psd[total_mask], freqs[total_mask])
            if total_mask.sum() >= 2 else 0.0
        )
        
        for band_name, (low, high) in self.bands.items():
            effective_high = min(high, self.sfreq / 2.0)
            mask = (freqs >= low) & (freqs < effective_high)
            power = self.trapz(psd[mask], freqs[mask]) if mask.sum() >= 2 else 0.0
            
            features[f"{prefix}_{band_name}_power"] = float(power)
            features[f"{prefix}_{band_name}_relative_power"] = float(
                power / total_power if total_power > 0 else 0.0
            )
            
            if not np.isfinite(features[f"{prefix}_{band_name}_power"]):
                features[f"{prefix}_{band_name}_power"] = 0.0
            if not np.isfinite(features[f"{prefix}_{band_name}_relative_power"]):
                features[f"{prefix}_{band_name}_relative_power"] = 0.0
                
        return features
