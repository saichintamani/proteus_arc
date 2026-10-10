import numpy as np
import mne

class EEGFilter:
    """
    Applies configurable bandpass filtering to EEG signals.
    """
    def __init__(self, l_freq: float = 0.5, h_freq: float = 45.0, method: str = 'iir'):
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.method = method
        
    def apply(self, data: np.ndarray, sfreq: float) -> np.ndarray:
        """
        Applies filter to continuous EEG data (channels, samples) using MNE.
        """
        if data.ndim != 2:
            raise ValueError("Data must be 2D (channels, samples).")
            
        # Ensure filter boundaries are valid for sfreq
        nyquist = sfreq / 2.0
        if self.h_freq >= nyquist:
            raise ValueError(f"High pass frequency {self.h_freq} must be less than Nyquist frequency {nyquist}.")
            
        if self.l_freq <= 0:
            raise ValueError(f"Low pass frequency {self.l_freq} must be > 0.")
            
        # Using mne.filter.filter_data
        filtered_data = mne.filter.filter_data(
            data.astype(np.float64), 
            sfreq=sfreq, 
            l_freq=self.l_freq, 
            h_freq=self.h_freq, 
            method=self.method,
            verbose='ERROR'
        )
        return filtered_data
