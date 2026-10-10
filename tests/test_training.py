import unittest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import tempfile
import os
from src.proteus_arc.models.cnn_1d import EEG1DCNN
from src.proteus_arc.models.training import EEGDataset, CNNTrainer

class TestTraining(unittest.TestCase):
    def setUp(self):
        self.batch_size = 4
        self.channels = 19
        self.time_steps = 500
        self.num_classes = 3
        
        self.model = EEG1DCNN(in_channels=self.channels, num_classes=self.num_classes, pool1_size=2, pool2_size=2)
        
        X = np.random.randn(8, self.channels, self.time_steps)
        y = np.random.randint(0, self.num_classes, size=(8,))
        g = np.array(["s1"] * 4 + ["s2"] * 4)
        
        self.dataset = EEGDataset(X, y, g)
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size)
        
    def test_backpropagation(self):
        # Save initial weights
        initial_weights = self.model.conv1.weight.clone()
        
        trainer = CNNTrainer(self.model, device="cpu", learning_rate=0.01)
        trainer.train_epoch(self.dataloader)
        
        # Check if weights updated
        self.assertFalse(torch.equal(initial_weights, self.model.conv1.weight))
        
    def test_checkpoint_saving(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "best_model.pth")
            trainer = CNNTrainer(self.model, device="cpu", patience=1)
            history = trainer.fit(self.dataloader, self.dataloader, epochs=2, save_path=save_path)
            
            self.assertTrue(os.path.exists(save_path))
            self.assertIn("train_loss", history)
            self.assertIn("val_loss", history)
            
            # Test loading
            loaded_model = EEG1DCNN(in_channels=self.channels, num_classes=self.num_classes, pool1_size=2, pool2_size=2)
            loaded_model.load_state_dict(torch.load(save_path, weights_only=True))

if __name__ == "__main__":
    unittest.main()
