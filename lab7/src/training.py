"""Training utilities for STSA-Net."""

import torch
from tqdm import tqdm
from collections import defaultdict


class TrainingHistory:
    """Track training and validation metrics."""
    
    def __init__(self):
        self.train_losses = []
        self.train_accs = []
        self.val_losses = []
        self.val_accs = []
    
    def update(self, train_loss, train_acc, val_loss, val_acc):
        """Update history with new metrics."""
        self.train_losses.append(train_loss)
        self.train_accs.append(train_acc)
        self.val_losses.append(val_loss)
        self.val_accs.append(val_acc)
    
    def get_best_val_acc(self):
        """Get the best validation accuracy."""
        return max(self.val_accs) if self.val_accs else 0.0


def train_epoch(model, loader, criterion, optimizer, device):
    """
    Train the model for one epoch.
    
    Args:
        model: The neural network model
        loader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer
        device: Device to run on (cuda/cpu)
    
    Returns:
        tuple: (average_loss, accuracy_percentage)
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for data, labels in tqdm(loader, desc='Training'):
        data, labels = data.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return total_loss / len(loader), 100 * correct / total


def validate(model, loader, criterion, device):
    """
    Validate the model.
    
    Args:
        model: The neural network model
        loader: DataLoader for validation data
        criterion: Loss function
        device: Device to run on (cuda/cpu)
    
    Returns:
        tuple: (average_loss, accuracy_percentage)
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, labels in tqdm(loader, desc='Validating'):
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return total_loss / len(loader), 100 * correct / total

