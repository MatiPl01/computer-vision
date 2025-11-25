"""Training utilities and functions"""

import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from collections import Counter
from tqdm import tqdm
from typing import Dict, Tuple, List

from .config import BATCH_SIZE


def setup_device(seed: int = 42) -> torch.device:
    """Setup device and set random seeds"""
    # Set random seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return device


def calculate_class_weights(labels: List[int], num_classes: int) -> Tuple[Dict[int, float], torch.Tensor]:
    """Calculate class weights for imbalanced dataset"""
    counter = Counter(labels)
    total = len(labels)
    
    # Calculate weights (inverse frequency)
    weights = {}
    for cls in range(num_classes):
        count = counter.get(cls, 1)
        weights[cls] = total / (num_classes * count)
    
    # Create tensor
    weights_tensor = torch.tensor([weights[i] for i in range(num_classes)], dtype=torch.float32)
    
    return weights, weights_tensor


def create_data_loaders(
    train_dataset,
    val_dataset,
    test_dataset,
    class_weights: Dict[int, float],
    batch_size: int = BATCH_SIZE
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create data loaders with weighted sampling for training"""
    # Create weighted sampler for training
    sample_weights = [class_weights[label] for label in train_dataset.labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return train_loader, val_loader, test_loader


def validate(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float, List[int], List[int]]:
    """Validate model on a dataset"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
    
    avg_loss = total_loss / len(data_loader)
    accuracy = 100.0 * correct / total
    
    return avg_loss, accuracy, all_preds, all_labels


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    num_epochs: int,
    device: torch.device,
    patience: int = 10,
    max_grad_norm: float = 1.0,
    warmup_epochs: int = 5
) -> Dict:
    """Train the model"""
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'best_val_acc': 0.0,
        'best_model_state': None,
        'best_epoch': 0
    }
    
    best_val_acc = 0.0
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Training]')
        for inputs, labels in train_pbar:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            optimizer.step()
            
            train_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
            
            train_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.0 * train_correct / train_total:.2f}%'
            })
        
        avg_train_loss = train_loss / len(train_loader)
        train_acc = 100.0 * train_correct / train_total
        
        # Validation phase
        val_loss, val_acc, _, _ = validate(model, val_loader, criterion, device)
        
        # Learning rate scheduling (skip warmup epochs)
        if epoch >= warmup_epochs:
            scheduler.step(val_loss)
        
        # Update history
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            history['best_val_acc'] = best_val_acc
            history['best_model_state'] = model.state_dict().copy()
            history['best_epoch'] = epoch + 1
            patience_counter = 0
            print(f'✓ New best validation accuracy: {val_acc:.2f}%')
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break
    
    return history
