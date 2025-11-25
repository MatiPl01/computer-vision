"""Loss functions for training"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BalancedFocalLoss(nn.Module):
    """Focal loss with class balancing"""
    
    def __init__(self, alpha: torch.Tensor, gamma: float = 2.0):
        """
        Args:
            alpha: Class weights tensor of shape (num_classes,)
            gamma: Focusing parameter
        """
        super(BalancedFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Logits of shape (B, num_classes)
            targets: Target labels of shape (B,)
        
        Returns:
            Loss value
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()
