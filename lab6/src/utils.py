"""Utility functions for visualization and analysis"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from typing import Dict, List

from .config import CLASS_NAMES


def visualize_model_architecture(model: torch.nn.Module, input_shape: tuple):
    """Print model architecture and parameter count"""
    print("Model Architecture:")
    print("=" * 60)
    print(model)
    print("=" * 60)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    # Set to evaluation mode to avoid BatchNorm issues with batch_size=1
    was_training = model.training
    model.eval()
    try:
        dummy_input = torch.randn(input_shape)
        with torch.no_grad():
            output = model(dummy_input)
        
        print(f"\nInput shape: {input_shape}")
        print(f"Output shape: {output.shape}")
    finally:
        # Restore original training state
        model.train(was_training)


def plot_training_curves(results: Dict, window_sizes: List[int]):
    """Plot training and validation curves"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    for window_size in window_sizes:
        if window_size not in results:
            continue
        
        history = results[window_size]
        epochs = range(1, len(history['train_loss']) + 1)
        
        # Loss curves
        axes[0].plot(epochs, history['train_loss'], label=f'Train (w={window_size})', linestyle='--')
        axes[0].plot(epochs, history['val_loss'], label=f'Val (w={window_size})', linestyle='-')
        
        # Accuracy curves
        axes[1].plot(epochs, history['train_acc'], label=f'Train (w={window_size})', linestyle='--')
        axes[1].plot(epochs, history['val_acc'], label=f'Val (w={window_size})', linestyle='-')
    
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_confusion_matrices(results: Dict, window_sizes: List[int], class_names: List[str] = CLASS_NAMES):
    """Plot confusion matrices for different window sizes"""
    n = len(window_sizes)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    
    if n == 1:
        axes = [axes]
    
    for idx, window_size in enumerate(window_sizes):
        if window_size not in results:
            continue
        
        test_labels = results[window_size]['test_labels']
        test_preds = results[window_size]['test_preds']
        
        cm = confusion_matrix(test_labels, test_preds)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt='.2f',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            ax=axes[idx]
        )
        axes[idx].set_title(f'Window Size = {window_size}\nTest Acc: {results[window_size]["test_acc"]:.2f}%')
        axes[idx].set_xlabel('Predicted')
        axes[idx].set_ylabel('True')
    
    plt.tight_layout()
    plt.show()


def print_summary(results: Dict, window_sizes: List[int], class_names: List[str] = CLASS_NAMES):
    """Print summary of results"""
    print("\n" + "=" * 60)
    print("SUMMARY OF RESULTS")
    print("=" * 60)
    print(f"{'Window Size':<15} {'Val Acc (%)':<15} {'Test Acc (%)':<15} {'Test Loss':<15}")
    print("-" * 60)
    
    best_window = None
    best_acc = 0.0
    
    for window_size in window_sizes:
        if window_size not in results:
            continue
        
        val_acc = results[window_size]['best_val_acc']
        test_acc = results[window_size]['test_acc']
        test_loss = results[window_size]['test_loss']
        
        print(f"{window_size:<15} {val_acc:<15.2f} {test_acc:<15.2f} {test_loss:<15.4f}")
        
        if test_acc > best_acc:
            best_acc = test_acc
            best_window = window_size
    
    print("=" * 60)
    if best_window is not None:
        print(f"✓ Best model: window_size={best_window} with test accuracy={best_acc:.2f}%")
    print()
