"""Base Conv2d model."""
import torch
from torch import Tensor, nn

from src.g2net.data import Batch
from src.g2net.modeling.base2d.layers import Conv2dBlock
from src.g2net.modeling.config import ModelConfig
from src.g2net.modeling.module import BaseLightningModule


class BaseModel2d(BaseLightningModule):
    """Simple Model."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.save_hyperparameters()

        self.encoder = nn.Sequential(
            Conv2dBlock(
                2,
                config.conv_channels[0],
                kernel_size=config.kernel_size,
                dropout=config.dropout,
            ),
            Conv2dBlock(
                config.conv_channels[0],
                config.conv_channels[1],
                kernel_size=config.kernel_size,
                dropout=config.dropout,
            ),
            Conv2dBlock(
                config.conv_channels[1],
                config.conv_channels[2],
                kernel_size=config.kernel_size,
                dropout=config.dropout,
            ),
            Conv2dBlock(
                config.conv_channels[2],
                config.conv_channels[3],
                kernel_size=config.kernel_size,
                dropout=config.dropout,
            ),
        )
        self.flatten = nn.Flatten()
        self.fc_1 = nn.Sequential(
            nn.Linear(8192, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
        )
        self.fc_2 = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
        )
        self.fc_3 = nn.Sequential(
            nn.Linear(64, 4),
            nn.ReLU(),
            nn.BatchNorm1d(4),
        )

        self.out_proj = nn.Linear(4, config.num_classes)

    def forward(self, batch: Batch) -> Tensor:
        """Forward Pass."""
        inputs = torch.concat((batch.h1_amplitude, batch.l1_amplitude), dim=1)
        x = self.encoder(inputs)
        x = self.flatten(x)
        x = self.fc_1(x)
        x = self.fc_2(x)
        x = self.fc_3(x)
        x = self.out_proj(x)
        return x
