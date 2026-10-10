import torch
import torch.nn as nn
import torch.nn.functional as F

class EEG1DCNN(nn.Module):
    """
    Compact 1D CNN for EEG Classification.
    Expected Input: (batch_size, in_channels, sequence_length)
    """
    def __init__(self, in_channels: int = 19, num_classes: int = 3, 
                 conv1_filters: int = 16, conv1_kernel: int = 31, conv1_stride: int = 1, pool1_size: int = 4,
                 conv2_filters: int = 32, conv2_kernel: int = 15, conv2_stride: int = 1, pool2_size: int = 4,
                 dropout: float = 0.5):
        super(EEG1DCNN, self).__init__()
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        
        # Block 1
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=conv1_filters, 
                               kernel_size=conv1_kernel, stride=conv1_stride, padding=conv1_kernel//2)
        self.bn1 = nn.BatchNorm1d(conv1_filters)
        self.pool1 = nn.MaxPool1d(kernel_size=pool1_size)
        
        # Block 2
        self.conv2 = nn.Conv1d(in_channels=conv1_filters, out_channels=conv2_filters, 
                               kernel_size=conv2_kernel, stride=conv2_stride, padding=conv2_kernel//2)
        self.bn2 = nn.BatchNorm1d(conv2_filters)
        self.pool2 = nn.MaxPool1d(kernel_size=pool2_size)
        
        # Adaptive Global Pooling (ensures consistent size for classification head regardless of input length)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Classification Head
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(conv2_filters, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input Validation
        if x.dim() != 3:
            raise ValueError(f"Expected input shape (batch, channels, time), but got {x.dim()}D tensor.")
        if x.shape[1] != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input channels, but got {x.shape[1]}.")
            
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        # Global Pooling and Head
        x = self.global_pool(x)
        x = x.squeeze(-1) # Remove temporal dimension (now size 1)
        
        x = self.dropout(x)
        out = self.fc(x)
        
        return out
