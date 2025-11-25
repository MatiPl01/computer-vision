"""3D CNN for Lip Sequence Recognition - Main Package"""

from .config import (
    DATA_DIR, TRAIN_SUBJECTS, VAL_SUBJECTS, TEST_SUBJECTS, WARMUP_EPOCHS,
    WINDOW_SIZES, BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, NUM_CLASSES, CLASS_NAMES,
    WEIGHT_DECAY, EARLY_STOPPING_PATIENCE, MAX_GRAD_NORM, FOCAL_LOSS_GAMMA
)
from .dataset import LipSequenceDataset, SequenceAugmentation
from .models import Conv3DNet, ImprovedConv3DNet
from .training import (
    setup_device, calculate_class_weights, create_data_loaders,
    train_model, validate
)
from .losses import BalancedFocalLoss
from .gradcam import GradCAM3D, visualize_gradcam
from .utils import (
    print_summary, plot_training_curves, plot_confusion_matrices,
    visualize_model_architecture
)

__all__ = [
    # Config
    'DATA_DIR', 'TRAIN_SUBJECTS', 'VAL_SUBJECTS', 'TEST_SUBJECTS', 'WARMUP_EPOCHS',
    'WINDOW_SIZES', 'BATCH_SIZE', 'NUM_EPOCHS', 'LEARNING_RATE', 'NUM_CLASSES', 'CLASS_NAMES',
    'WEIGHT_DECAY', 'EARLY_STOPPING_PATIENCE', 'MAX_GRAD_NORM', 'FOCAL_LOSS_GAMMA',
    # Dataset
    'LipSequenceDataset', 'SequenceAugmentation',
    # Models
    'Conv3DNet', 'ImprovedConv3DNet',
    # Training
    'setup_device', 'calculate_class_weights', 'create_data_loaders', 'train_model', 'validate',
    # Losses
    'BalancedFocalLoss',
    # GradCAM
    'GradCAM3D', 'visualize_gradcam',
    # Utils
    'print_summary', 'plot_training_curves', 'plot_confusion_matrices', 'visualize_model_architecture',
]
