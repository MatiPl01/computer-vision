"""Configuration constants for the lip sequence recognition project"""

from pathlib import Path

# Data directory
DATA_DIR = Path('data')

# Subject splits (based on README.md)
TRAIN_SUBJECTS = ['p1', 'p1_h', 'p2', 'p2_h', 'p3', 'p3_h', 'p4', 'p4_h', 'p8', 'p8_h', 'p9', 'p9_h']
VAL_SUBJECTS = ['p6', 'p6_h']
TEST_SUBJECTS = ['p7', 'p7_h', 'p10', 'p10_h']

# Model hyperparameters
WINDOW_SIZES = [5, 7, 9]  # Moving window sizes to experiment with
BATCH_SIZE = 32
NUM_EPOCHS = 60
LEARNING_RATE = 0.0005  # Reduced from 0.001 to reduce overfitting
WEIGHT_DECAY = 1e-4  # Increased from 1e-5 for stronger regularization
WARMUP_EPOCHS = 5

# Training parameters
EARLY_STOPPING_PATIENCE = 10
MAX_GRAD_NORM = 1.0
FOCAL_LOSS_GAMMA = 2.0

# Dataset parameters
NUM_CLASSES = 6
CLASS_NAMES = ['Silence/Neutral', '/A/', '/I/', '/U/', '/E/', '/O/']
