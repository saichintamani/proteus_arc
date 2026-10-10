import unittest
import numpy as np
from src.proteus_arc.preprocessing.filtering import EEGFilter

class TestFiltering(unittest.TestCase):
    def setUp(self):
        self.sfreq = 500.0
        self.n_channels = 2
        self.duration_sec = 10.0
        self.n_samples = int(self.sfreq * self.duration_sec)
        
        # Create a synthetic signal with a low freq (2Hz) and high freq (100Hz) component
        t = np.arange(self.n_samples) / self.sfreq
        self.sig_2hz = np.sin(2 * np.pi * 2 * t)
        self.sig_100hz = np.sin(2 * np.pi * 100 * t)
        self.data = np.stack([self.sig_2hz + self.sig_100hz, self.sig_2hz + self.sig_100hz])
        
    def test_filter_boundaries_validity(self):
        # Invalid high pass (above Nyquist)
        with self.assertRaises(ValueError):
            EEGFilter(l_freq=0.5, h_freq=300.0).apply(self.data, self.sfreq)
            
        # Invalid low pass
        with self.assertRaises(ValueError):
            EEGFilter(l_freq=-1.0, h_freq=45.0).apply(self.data, self.sfreq)
            
    def test_filtering_attenuates_high_freq(self):
        filter_obj = EEGFilter(l_freq=0.5, h_freq=45.0, method='iir')
        filtered_data = filter_obj.apply(self.data, self.sfreq)
        
        # Original data has high frequency component. 
        # After filtering < 45Hz, the 100Hz component should be heavily attenuated.
        orig_power = np.mean(self.data**2)
        filtered_power = np.mean(filtered_data**2)
        
        self.assertTrue(filtered_power < orig_power)
        
if __name__ == "__main__":
    unittest.main()
