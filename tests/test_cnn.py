import unittest
import torch
from src.proteus_arc.models.cnn_1d import EEG1DCNN

class TestCNNArchitecture(unittest.TestCase):
    def setUp(self):
        self.batch_size = 4
        self.channels = 19
        self.time_steps = 15000 # 30s at 500Hz
        self.num_classes = 3
        
        self.model = EEG1DCNN(
            in_channels=self.channels,
            num_classes=self.num_classes,
            conv1_filters=8, conv1_kernel=5,
            conv2_filters=16, conv2_kernel=5
        )
        
    def test_forward_pass_shape(self):
        # Create synthetic input
        x = torch.randn(self.batch_size, self.channels, self.time_steps)
        out = self.model(x)
        
        # Expected output: (batch_size, num_classes)
        self.assertEqual(out.shape, (self.batch_size, self.num_classes))
        
        # Values should be finite
        self.assertTrue(torch.isfinite(out).all())
        
    def test_invalid_input_dim(self):
        # 4D tensor instead of 3D
        x = torch.randn(self.batch_size, self.channels, self.time_steps, 2)
        with self.assertRaises(ValueError):
            self.model(x)
            
    def test_invalid_channels(self):
        x = torch.randn(self.batch_size, self.channels + 1, self.time_steps)
        with self.assertRaises(ValueError):
            self.model(x)

if __name__ == "__main__":
    unittest.main()
