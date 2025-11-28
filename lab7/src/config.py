"""Configuration parameters for STSA-Net training."""


class Config:
    """Model and training configuration."""
    
    # Model hyperparameters
    T = 300  # Number of time frames
    NUM_JOINTS = 17  # Number of skeleton joints (COCO format: 17 joints)
    JOINT_DIM = 3  # Joint dimensions (x, y, z coordinates)
    H = 256  # Hidden dimension
    NUM_CLASSES = 5  # Number of gymnastic exercise classes
    NUM_HEADS = 8  # Number of attention heads
    DROPOUT = 0.1  # Dropout rate
    FFN_MULT = 4  # Feed-forward network multiplier
    NUM_BLOCKS = 8  # Number of STSA blocks (as mentioned in the paper)
    
    # Training hyperparameters
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 50
    WEIGHT_DECAY = 1e-5
    
    # Dataset difficulty (for synthetic data)
    NOISE_LEVEL = 0.05  # Increased noise for more realistic data
    PATTERN_VARIATION = True  # Enable pattern variations
    
    # Dataset parameters
    TRAIN_SAMPLES = 800
    VAL_SAMPLES = 100
    TEST_SAMPLES = 100
    
    # Exercise class names
    CLASS_NAMES = ['Jumping Jacks', 'Squats', 'Arm Circles', 'Lunges', 'Push-ups']

