"""3D Convolutional Neural Network models for lip sequence recognition"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv3DNet(nn.Module):
    """3D CNN with temporal attention for lip sequence recognition"""
    
    def __init__(self, num_classes: int = 6, window_size: int = 9):
        """
        Args:
            num_classes: Number of output classes
            window_size: Temporal window size
        """
        super(Conv3DNet, self).__init__()
        
        # Conv block 1
        self.conv1 = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Dropout3d(0.15),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        )
        
        # Conv block 2
        self.conv2 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Dropout3d(0.25),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        )
        
        # Conv block 3
        self.conv3 = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.Dropout3d(0.35),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        )
        
        # Conv block 4
        self.conv4 = nn.Sequential(
            nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),
            nn.Dropout3d(0.45),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        )
        
        # Temporal attention
        self.temporal_attention = nn.Sequential(
            nn.Conv3d(512, 256, kernel_size=(3, 1, 1), padding=(1, 0, 0)),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 512, kernel_size=(3, 1, 1), padding=(1, 0, 0)),
            nn.Sigmoid()
        )
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        # Fully connected layers
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256)
        )
        
        self.fc2 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True)
        )
        
        self.classifier = nn.Linear(128, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, C, T, H, W)
        
        Returns:
            Output logits of shape (B, num_classes)
        """
        # Convolutional blocks
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        
        # Temporal attention
        attention = self.temporal_attention(x)
        x = x * attention
        
        # Global pooling
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.classifier(x)
        
        return x


class ImprovedConv3DNet(nn.Module):
    """Improved 3D CNN with residual connections"""
    
    def __init__(self, num_classes: int = 6, window_size: int = 9):
        """
        Args:
            num_classes: Number of output classes
            window_size: Temporal window size
        """
        super(ImprovedConv3DNet, self).__init__()
        
        # Initial conv block
        self.conv1 = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Dropout3d(0.2),  # Increased from 0.15
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        )
        
        # Residual blocks
        self.res_block1 = self._make_residual_block(64, 128)
        self.res_block2 = self._make_residual_block(128, 256)
        self.res_block3 = self._make_residual_block(256, 512)
        
        # Temporal attention
        self.temporal_attention = nn.Sequential(
            nn.Conv3d(512, 256, kernel_size=(3, 1, 1), padding=(1, 0, 0)),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 512, kernel_size=(3, 1, 1), padding=(1, 0, 0)),
            nn.Sigmoid()
        )
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        # Fully connected layers
        self.fc1 = nn.Sequential(
            nn.Dropout(0.6),  # Increased from 0.5
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256)
        )
        
        self.fc2 = nn.Sequential(
            nn.Dropout(0.6),  # Increased from 0.5
            nn.Linear(256, 128),
            nn.ReLU(inplace=True)
        )
        
        self.classifier = nn.Linear(128, num_classes)
    
    def _make_residual_block(self, in_channels: int, out_channels: int) -> nn.Module:
        """Create a residual block"""
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout3d(0.3),  # Increased from 0.25
            # Use (1, 2, 2) for temporal dimension to avoid reducing it when it's already small
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, C, T, H, W)
        
        Returns:
            Output logits of shape (B, num_classes)
        """
        # Initial conv
        x = self.conv1(x)
        
        # Residual blocks
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        
        # Temporal attention
        attention = self.temporal_attention(x)
        x = x * attention
        
        # Global pooling
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.classifier(x)
        
        return x

