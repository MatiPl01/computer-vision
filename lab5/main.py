from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset


# -----------------------------
#  Graph definition utilities
# -----------------------------

JOINT_NAMES = [
    "Pelvis",
    "LeftShoulder",
    "LeftElbow",
    "LeftWrist",
    "RightShoulder",
    "RightElbow",
    "RightWrist",
]

GRAPH_EDGES = [
    (0, 1),
    (1, 2),
    (2, 3),
    (0, 4),
    (4, 5),
    (5, 6),
    (1, 4),  # connect shoulders
]


def build_normalised_adjacency(
    num_joints: int, edges: Sequence[Tuple[int, int]]
) -> torch.Tensor:
    adjacency = torch.zeros((num_joints, num_joints), dtype=torch.float32)
    for i, j in edges:
        adjacency[i, j] = 1.0
        adjacency[j, i] = 1.0
    adjacency += torch.eye(num_joints)

    degree = adjacency.sum(dim=1)
    degree_inv_sqrt = torch.pow(degree, -0.5)
    degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0.0
    d_mat = torch.diag(degree_inv_sqrt)
    return d_mat @ adjacency @ d_mat


# -----------------------------
#    Dataset / preprocessing
# -----------------------------

MEDIA_PIPE_INDEX_MAP = {
    "Pelvis": ("hips_midpoint",),
    "LeftShoulder": (11,),
    "LeftElbow": (13,),
    "LeftWrist": (15,),
    "RightShoulder": (12,),
    "RightElbow": (14,),
    "RightWrist": (16,),
}


def _extract_hips_midpoint(landmarks: np.ndarray) -> np.ndarray:
    left = landmarks[:, 23, :2]
    right = landmarks[:, 24, :2]
    return (left + right) / 2.0


def load_mediapipe_sequence(path: Path) -> np.ndarray:
    data = np.load(path)
    if isinstance(data, np.lib.npyio.NpzFile):
        if "landmarks" in data:
            landmarks = data["landmarks"]
        else:
            landmarks = data[data.files[0]]
    else:
        landmarks = data

    if landmarks.ndim != 3 or landmarks.shape[1] < 33 or landmarks.shape[2] < 2:
        raise ValueError(
            f"Unexpected MediaPipe landmark shape in {path}: {landmarks.shape}"
        )

    hips = _extract_hips_midpoint(landmarks)
    joint_coords = []
    for joint in JOINT_NAMES:
        indices = MEDIA_PIPE_INDEX_MAP[joint]
        if indices == ("hips_midpoint",):
            joint_coords.append(hips)
        else:
            joint_coords.append(landmarks[:, indices[0], :2])
    stacked = np.stack(joint_coords, axis=1)  # (T, V, 2)
    return stacked.astype(np.float32)


def load_csv_sequence(path: Path) -> np.ndarray:
    df = pd.read_csv(path, skiprows=2, header=None, dtype=np.float32)
    df = df.dropna(axis=1, how="all").reset_index(drop=True)
    df.columns = range(df.shape[1])
    num_joints = len(JOINT_NAMES)
    expected_cols = num_joints * 2
    if df.shape[1] != expected_cols:
        raise ValueError(
            f"File {path} has {df.shape[1]} columns after cleaning, expected {expected_cols}."
        )
    data = df.to_numpy(dtype=np.float32).reshape(-1, num_joints, 2)
    return data


def normalise_sequence(
    sequence: np.ndarray, center: bool = True, scale: bool = True
) -> np.ndarray:
    seq = sequence.copy()
    if center:
        root = seq[:, :1, :]
        seq = seq - root
    if scale:
        scale_value = np.maximum(np.linalg.norm(seq, axis=-1).max(), 1e-6)
        seq = seq / scale_value
    return seq


class SkeletonSequenceDataset(Dataset):
    def __init__(
        self,
        root_dir: Path,
        label_map: Dict[str, int],
        file_extensions: Iterable[str] = (".csv", ".npz", ".npy"),
        cache: bool = True,
        max_per_class: int | None = None,
        center: bool = True,
        scale: bool = False,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.label_map = label_map
        self.file_extensions = tuple(ext.lower() for ext in file_extensions)
        self.center = center
        self.scale = scale
        self.cache = cache

        self.samples: List[Tuple[Path, int]] = []
        self._cached_sequences: List[torch.Tensor] = []

        for label_name, label_idx in sorted(self.label_map.items()):
            class_dir = self.root_dir / label_name
            if not class_dir.exists():
                continue
            class_samples: List[Tuple[Path, int]] = []
            for file in sorted(class_dir.rglob("*")):
                if file.suffix.lower() in self.file_extensions:
                    class_samples.append((file, label_idx))
                    if (
                        max_per_class is not None
                        and len(class_samples) >= max_per_class
                    ):
                        break
            self.samples.extend(class_samples)

        if cache:
            for file_path, _ in self.samples:
                self._cached_sequences.append(self._load_and_prepare(file_path))

    def _load_and_prepare(self, path: Path) -> torch.Tensor:
        if path.suffix.lower() == ".csv":
            sequence = load_csv_sequence(path)
        else:
            sequence = load_mediapipe_sequence(path)
        sequence = normalise_sequence(sequence, center=self.center, scale=self.scale)
        sequence = sequence.transpose(2, 0, 1)  # (C=2, T, V)
        return torch.from_numpy(sequence).float()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        file_path, label = self.samples[idx]
        if self.cache:
            sequence = self._cached_sequences[idx]
        else:
            sequence = self._load_and_prepare(file_path)
        return sequence, torch.tensor(label, dtype=torch.long)


# -----------------------------
#         Model blocks
# -----------------------------


class GraphConvLayer(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, adjacency: torch.Tensor
    ) -> None:
        super().__init__()
        self.register_buffer("adjacency", adjacency)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.einsum("vw,bctw->bctv", self.adjacency, x)
        return self.conv(x)


class STGCNBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        adjacency: torch.Tensor,
        stride: int = 1,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.gcn = GraphConvLayer(in_channels, out_channels, adjacency)
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=(3, 1),
                stride=(stride, 1),
                padding=(1, 0),
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout),
        )

        if stride != 1 or in_channels != out_channels:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=(1, 1), stride=(stride, 1)
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.residual = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual(x)
        x = self.gcn(x)
        x = self.tcn(x)
        x = x + residual
        return F.relu(x)


class STGCN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_joints: int,
        num_classes: int,
        adjacency: torch.Tensor,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.data_bn = nn.BatchNorm1d(in_channels * num_joints)
        self.adjacency = adjacency

        self.blocks = nn.ModuleList(
            [
                STGCNBlock(in_channels, 16, adjacency, stride=1, dropout=dropout),
                STGCNBlock(16, 32, adjacency, stride=1, dropout=dropout),
            ]
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t, v = x.shape

        x = x.permute(0, 3, 1, 2).contiguous().view(b, v * c, t)
        x = self.data_bn(x)
        x = x.view(b, v, c, t).permute(0, 2, 3, 1).contiguous()

        for block in self.blocks:
            x = block(x)

        return self.head(x)


# -----------------------------
#       Training helpers
# -----------------------------


@dataclass
class TrainingConfig:
    train_dir: Path = Path("train/skeleton")
    val_dir: Path = Path("test/skeleton")
    epochs: int = 50
    batch_size: int = 16
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    dropout: float = 0.35
    num_workers: int = 0
    seed: int = 42
    device: str | None = None
    max_train_per_class: int | None = None
    max_val_per_class: int | None = None
    cache_dataset: bool = True
    center: bool = True
    scale: bool = False


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for sequences, labels in loader:
        sequences = sequences.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(sequences)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

    return total_loss / total_samples, total_correct / total_samples


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    all_preds: List[int] = []
    all_targets: List[int] = []

    with torch.no_grad():
        for sequences, labels in loader:
            sequences = sequences.to(device)
            labels = labels.to(device)

            logits = model(sequences)
            loss = F.cross_entropy(logits, labels)

            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(labels.cpu().tolist())

    return (
        total_loss / total_samples,
        total_correct / total_samples,
        np.array(all_targets),
        np.array(all_preds),
    )


def build_dataloaders(
    config: TrainingConfig,
) -> Tuple[DataLoader, DataLoader, Dict[str, int]]:
    label_map = {
        name: idx
        for idx, name in enumerate(
            sorted(
                [p.name for p in Path(config.train_dir).iterdir() if p.is_dir()],
                key=str.lower,
            )
        )
    }

    device_hint = (
        config.device or ("cuda" if torch.cuda.is_available() else "cpu")
    ).lower()
    use_pin_memory = "cuda" in device_hint

    train_dataset = SkeletonSequenceDataset(
        config.train_dir,
        label_map,
        cache=config.cache_dataset,
        max_per_class=config.max_train_per_class,
        center=config.center,
        scale=config.scale,
    )
    val_dataset = SkeletonSequenceDataset(
        config.val_dir,
        label_map,
        cache=config.cache_dataset,
        max_per_class=config.max_val_per_class,
        center=config.center,
        scale=config.scale,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=use_pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=use_pin_memory,
    )
    return train_loader, val_loader, label_map


def run_training(config: TrainingConfig) -> None:
    set_seed(config.seed)
    device = torch.device(
        config.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    train_loader, val_loader, label_map = build_dataloaders(config)
    adjacency = build_normalised_adjacency(len(JOINT_NAMES), GRAPH_EDGES).to(device)

    model = STGCN(
        in_channels=2,
        num_joints=len(JOINT_NAMES),
        num_classes=len(label_map),
        adjacency=adjacency,
        dropout=config.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )

    print(
        f"Training on {len(train_loader.dataset)} sequences; validating on {len(val_loader.dataset)} sequences"
    )
    for epoch in range(1, config.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc, targets, preds = evaluate(model, val_loader, device)
        scheduler.step(val_acc)

        print(
            f"Epoch {epoch:03d} | "
            f"train loss {train_loss:.4f} acc {train_acc:.3f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.3f}"
        )

    print("\nValidation classification report:")
    target_names = [name for name, _ in sorted(label_map.items(), key=lambda x: x[1])]
    print(
        classification_report(
            targets,
            preds,
            target_names=target_names,
            zero_division=0,
        )
    )
    print("Confusion matrix:\n", confusion_matrix(targets, preds))


def parse_args() -> TrainingConfig:
    parser = argparse.ArgumentParser(
        description="ST-GCN skeleton-based action recognition"
    )
    parser.add_argument(
        "--train-dir",
        type=Path,
        default=TrainingConfig.train_dir,
        help="Directory with training skeleton sequences",
    )
    parser.add_argument(
        "--val-dir",
        type=Path,
        default=TrainingConfig.val_dir,
        help="Directory with validation/test skeleton sequences",
    )
    parser.add_argument("--epochs", type=int, default=TrainingConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=TrainingConfig.batch_size)
    parser.add_argument(
        "--learning-rate", type=float, default=TrainingConfig.learning_rate
    )
    parser.add_argument(
        "--weight-decay", type=float, default=TrainingConfig.weight_decay
    )
    parser.add_argument("--dropout", type=float, default=TrainingConfig.dropout)
    parser.add_argument("--num-workers", type=int, default=TrainingConfig.num_workers)
    parser.add_argument("--seed", type=int, default=TrainingConfig.seed)
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device, e.g. 'cpu' or 'cuda:0'",
    )
    parser.add_argument(
        "--max-train-per-class",
        type=int,
        default=None,
        help="Optionally limit how many sequences per class are used for training (useful for quick tests).",
    )
    parser.add_argument(
        "--max-val-per-class",
        type=int,
        default=None,
        help="Optionally limit how many sequences per class are used for validation.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable dataset caching (default caches all sequences in memory).",
    )
    parser.add_argument(
        "--no-center",
        action="store_true",
        help="Disable pelvis-centering before feeding sequences to the network.",
    )
    parser.add_argument(
        "--scale",
        action="store_true",
        help="Scale each sequence by its maximum joint displacement (disabled by default).",
    )
    args = parser.parse_args()
    return TrainingConfig(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        num_workers=args.num_workers,
        seed=args.seed,
        device=args.device,
        max_train_per_class=args.max_train_per_class,
        max_val_per_class=args.max_val_per_class,
        cache_dataset=not args.no_cache,
        center=not args.no_center,
        scale=args.scale,
    )


if __name__ == "__main__":
    cfg = parse_args()
    run_training(cfg)
