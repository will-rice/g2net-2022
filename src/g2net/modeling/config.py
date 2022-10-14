"""Model configuration."""
import os
from typing import NamedTuple


class ModelConfig(NamedTuple):
    """Configuration for the model."""

    # Training
    seed: int = 42
    batch_size: int = 16
    max_epochs: int = 10
    learning_rate: float = 4e-4
    swa_learning_rate: float = 1e-5
    weight_decay: float = 1e-6
    gamma: float = 0.9997
    lr_min: float = 1e-6
    warmup_t: int = 1.0
    clip_norm: float = 1000.0

    # Data
    augment_p = 1.0
    num_workers: int = os.cpu_count()
    num_folds: int = 5

    # All Models
    in_channels: int = 2
    num_classes: int = 1
    pos_weight: float = 1.0
    dropout: float = 0.2
    subsample_kernel_size: int = 1
    subsample_stride: int = 4

    # Pretrained models only
    base_model: str = "tf_efficientnet_b5_ns"
    img_size: int = 320

    # Base model only
    kernel_size: int = 16
    conv_channels: tuple = (32, 64, 128, 256)
