"""Grad-CAM visualization for 3D CNNs"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

from .config import CLASS_NAMES


class GradCAM3D:
    """Grad-CAM for 3D convolutional networks"""
    
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        """
        Args:
            model: The model to visualize
            target_layer: The target convolutional layer
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        """Save activation maps"""
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        """Save gradients"""
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_tensor: torch.Tensor, class_idx: int = None) -> np.ndarray:
        """
        Generate class activation map
        
        Args:
            input_tensor: Input tensor of shape (1, C, T, H, W)
            class_idx: Class index to visualize (None for predicted class)
        
        Returns:
            CAM heatmap of shape (T, H, W)
        """
        self.model.eval()
        
        # Forward pass
        output = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        output[0, class_idx].backward()
        
        # Get gradients and activations
        gradients = self.gradients[0]  # (C, T, H, W)
        activations = self.activations[0]  # (C, T, H, W)
        
        # Compute weights (global average pooling of gradients)
        weights = gradients.mean(dim=(2, 3), keepdim=True)  # (C, T, 1, 1)
        
        # Weighted combination of activation maps
        cam = (weights * activations).sum(dim=0)  # (T, H, W)
        
        # Apply ReLU
        cam = F.relu(cam)
        
        # Normalize
        cam = cam.cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam


def visualize_gradcam(
    model: torch.nn.Module,
    gradcam: GradCAM3D,
    input_tensor: torch.Tensor,
    true_label: int,
    window_size: int,
    class_names: list = CLASS_NAMES
):
    """Visualize Grad-CAM heatmaps for a sequence"""
    model.eval()
    
    # Get prediction
    with torch.no_grad():
        output = model(input_tensor)
        pred_label = output.argmax(dim=1).item()
        pred_prob = F.softmax(output, dim=1)[0, pred_label].item()
    
    # Generate CAM
    cam = gradcam.generate_cam(input_tensor, class_idx=pred_label)
    
    # Get input sequence
    input_seq = input_tensor[0].cpu().numpy()  # (C, T, H, W)
    input_seq = np.transpose(input_seq, (1, 2, 3, 0))  # (T, H, W, C)
    input_seq = (input_seq * 255).astype(np.uint8)
    
    # Create visualization
    fig, axes = plt.subplots(2, window_size, figsize=(window_size * 2, 4))
    fig.suptitle(
        f'Grad-CAM Visualization\n'
        f'True: {class_names[true_label]}, '
        f'Pred: {class_names[pred_label]} ({pred_prob:.2%})',
        fontsize=12
    )
    
    for t in range(window_size):
        # Original image
        axes[0, t].imshow(input_seq[t])
        axes[0, t].set_title(f'Frame {t+1}')
        axes[0, t].axis('off')
        
        # Heatmap
        heatmap = cam[t]
        axes[1, t].imshow(input_seq[t])
        axes[1, t].imshow(heatmap, alpha=0.5, cmap='jet')
        axes[1, t].set_title(f'CAM {t+1}')
        axes[1, t].axis('off')
    
    plt.tight_layout()
    plt.show()
