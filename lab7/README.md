# STSA-Net for Gymnastic Exercise Recognition

This project implements a transformer-based network using STSA-Net (Spatio-Temporal Self-Attention Network) blocks for recognizing gymnastic exercises from skeleton data.

## Project Structure

```
lab7/
├── STSANet.py              # STSA-Net model implementation
├── main.ipynb              # Main training and evaluation notebook
├── requirements.txt        # Python dependencies
├── src/                    # Source code modules
│   ├── __init__.py        # Package initialization
│   ├── config.py          # Configuration parameters
│   ├── dataset.py         # Synthetic dataset implementation
│   ├── training.py        # Training utilities
│   └── utils.py           # Visualization utilities
└── README.md              # This file
```

## Setup

### 1. Create Virtual Environment

**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**Linux/Mac:**
```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Notebook

```bash
jupyter notebook main.ipynb
```

## Model Architecture

The STSA-Net architecture consists of:

1. **Spatial Attention Block**: Captures relationships between joints within each frame
2. **Temporal Attention Block**: Captures temporal dependencies across frames for each joint
3. **STSA Block**: Combines spatial and temporal attention with feed-forward networks
4. **Positional Embeddings**: Learnable spatial and temporal position encodings

### Input Format

The model expects skeleton sequences of shape `(B, T, J, D)` where:
- `B`: Batch size
- `T`: Number of time frames (300)
- `J`: Number of joints (17)
- `D`: Joint dimensions (3 for x, y, z coordinates)

## Dataset

The project includes a synthetic dataset generator (`src/dataset.py`) that creates skeleton sequences for 5 gymnastic exercises:
1. Jumping Jacks
2. Squats
3. Arm Circles
4. Lunges
5. Push-ups

Each exercise has distinct motion patterns that the model learns to recognize. The dataset generates realistic skeleton sequences with temporal dynamics specific to each exercise type.

## Code Organization

The code is organized into modular components:

- **`src/config.py`**: Centralized configuration management
- **`src/dataset.py`**: Synthetic dataset generator
- **`src/training.py`**: Training and validation functions
- **`src/utils.py`**: Visualization utilities

This modular structure makes it easy to:
- Modify hyperparameters in one place
- Swap datasets
- Reuse training functions
- Extend functionality

## Training

The training pipeline includes:
- Data loading and preprocessing
- Model initialization
- Training with validation (using `src/training.py`)
- Learning rate scheduling
- Model checkpointing
- Evaluation metrics and visualizations

## Results

The model achieves good performance on the synthetic gymnastic exercise dataset, demonstrating the effectiveness of the STSA-Net architecture for skeleton-based action recognition.

## References

- Continuous Hand Gesture Recognition for Human-Robot Collaborative Assembly
- STSA-Net: Spatio-Temporal Self-Attention Network for Action Recognition

