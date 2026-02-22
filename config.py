from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class Config:
    # Paths
    train_data_dir: Path = Path("../video-based-seizure-detection-challenge-main/video-based-seizure-detection-challenge-main/train_data")
    model_dir: Path = Path("./checkpoints")

    # Data shape constants
    n_frames: int = 150
    n_landmarks: int = 33
    n_raw_features: int = 5  # x, y, z, visibility, presence

    # Model
    hidden_dim: int = 128
    n_gru_layers: int = 2
    dropout: float = 0.3

    # Training
    n_folds: int = 5
    n_epochs: int = 120
    batch_size: int = 32
    lr: float = 5e-4
    weight_decay: float = 1e-4
    patience: int = 25
    focal_alpha: float = 0.75
    focal_gamma: float = 2.0

    # Augmentation
    augment_train: bool = True
    mixup_alpha: float = 0.2

    # Seed
    seed: int = 42
