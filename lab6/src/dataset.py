"""Dataset classes for lip sequence recognition"""

import os
import random
from pathlib import Path
from collections import Counter
from typing import Optional, List, Tuple, Callable

import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import cv2


class SequenceAugmentation:
    """Augmentation for image sequences (temporal-aware)"""
    
    def __init__(self, p: float = 0.5):
        """
        Args:
            p: Probability of applying augmentation (increased from 0.3 to 0.5)
        """
        self.p = p
    
    def __call__(self, sequence: torch.Tensor) -> torch.Tensor:
        """
        Apply augmentation to a sequence of images
        
        Args:
            sequence: Tensor of shape (C, T, H, W)
        
        Returns:
            Augmented sequence
        """
        # Temporal augmentation: randomly reverse sequence (more aggressive)
        if random.random() < 0.4:  # Increased from 0.3
            sequence = torch.flip(sequence, dims=[1])
        
        # Spatial augmentation: apply same transform to all frames
        if random.random() < 0.6:  # Increased from 0.5
            # Random horizontal flip (applied to all frames)
            if random.random() < 0.5:
                sequence = torch.flip(sequence, dims=[3])
        
        # Color augmentation: apply to all frames (more aggressive)
        if random.random() < 0.6:  # Increased from 0.4
            # Random brightness adjustment (wider range)
            brightness_factor = random.uniform(0.7, 1.3)  # Increased from 0.8-1.2
            sequence = sequence * brightness_factor
            sequence = torch.clamp(sequence, 0, 1)
        
        if random.random() < 0.5:  # Increased from 0.3
            # Random contrast adjustment (wider range)
            contrast_factor = random.uniform(0.8, 1.2)  # Increased from 0.9-1.1
            mean = sequence.mean()
            sequence = (sequence - mean) * contrast_factor + mean
            sequence = torch.clamp(sequence, 0, 1)
        
        # Add Gaussian noise (new augmentation)
        if random.random() < 0.3:
            noise = torch.randn_like(sequence) * 0.05
            sequence = sequence + noise
            sequence = torch.clamp(sequence, 0, 1)
        
        return sequence


class LipSequenceDataset(Dataset):
    """Dataset for lip sequence recognition with moving window approach"""
    
    def __init__(
        self,
        data_dir: Path,
        subjects: List[str],
        window_size: int = 9,
        stride: Optional[int] = None,
        transform: Optional[Callable] = None,
        mode: str = 'train'
    ):
        """
        Args:
            data_dir: Path to data directory
            subjects: List of subject IDs to include
            window_size: Size of the temporal window
            stride: Stride for moving window (default: window_size // 2)
            transform: Optional augmentation transform
            mode: 'train', 'val', or 'test'
        """
        self.data_dir = Path(data_dir)
        self.subjects = subjects
        self.window_size = window_size
        self.stride = stride if stride is not None else window_size // 2
        self.transform = transform
        self.mode = mode
        
        self.sequences = []
        self.labels = []
        
        self._load_sequences()
    
    def _load_sequences(self):
        """Load all sequences from the dataset"""
        lips_dir = self.data_dir / 'lips'
        
        for subject in self.subjects:
            subject_dir = lips_dir / subject
            if not subject_dir.exists():
                continue
            
            # Get all frame files and sort them
            frame_files = sorted(subject_dir.glob('frame_*.png'))
            if not frame_files:
                continue
            
            # Extract frame indices and labels
            frames_data = []
            for frame_file in frame_files:
                # Parse filename: frame_XXXX_YY.png
                parts = frame_file.stem.split('_')
                if len(parts) >= 3:
                    frame_idx = int(parts[1])
                    label = int(parts[2])
                    frames_data.append((frame_idx, label, frame_file))
            
            # Sort by frame index
            frames_data.sort(key=lambda x: x[0])
            
            # Create sequences using moving window
            for i in range(0, len(frames_data) - self.window_size + 1, self.stride):
                window_frames = frames_data[i:i + self.window_size]
                
                # Determine label by majority vote
                window_labels = [f[1] for f in window_frames]
                label = max(set(window_labels), key=window_labels.count)
                
                # Store sequence info
                self.sequences.append([f[2] for f in window_frames])
                self.labels.append(label)
        
        print(f"Created {len(self.sequences)} sequences for {self.mode} set (window_size={self.window_size})")
        print(f"Class distribution: {Counter(self.labels)}")
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get a sequence and its label"""
        frame_files = self.sequences[idx]
        label = self.labels[idx]
        
        # Load images
        images = []
        for frame_file in frame_files:
            img = Image.open(frame_file).convert('RGB')
            img_array = np.array(img, dtype=np.float32) / 255.0
            images.append(img_array)
        
        # Stack into sequence: (T, H, W, C) -> (C, T, H, W)
        sequence = np.stack(images, axis=0)
        sequence = np.transpose(sequence, (3, 0, 1, 2))
        sequence = torch.from_numpy(sequence).float()
        
        # Apply augmentation if provided
        if self.transform is not None:
            sequence = self.transform(sequence)
        
        return sequence, label
