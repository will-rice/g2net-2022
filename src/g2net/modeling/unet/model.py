"""Efficientnet classification model with denoising UNet frontend."""
import torch
from efficientnet_pytorch import EfficientNet
from pytorch_lightning import LightningModule
from torch import Tensor, nn
from torchvision import transforms as T

from src.g2net.data import Batch
from src.g2net.modeling.config import ModelConfig
from src.g2net.modeling.layers import GlobalAvgPool2d
from src.g2net.modeling.module import BaseLightningModule
from src.g2net.modeling.unet.layers import UNet


class UNetModel(BaseLightningModule):
    """Simple Model."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.save_hyperparameters()

        self.output_size = {
            "efficientnet-b0": 1280,
            "efficientnet-b1": 1280,
            "efficientnet-b2": 1408,
            "efficientnet-b3": 1536,
            "efficientnet-b4": 1792,
            "efficientnet-b5": 2048,
            "efficientnet-b6": 2304,
            "efficientnet-b7": 2560,
        }
        self.frontend = UNet(
            in_channels=config.in_channels, out_channels=config.in_channels
        )
        self.encoder = EfficientNet.from_pretrained(
            config.base_model, in_channels=config.in_channels
        )
        self.pool = GlobalAvgPool2d()
        self.out_proj = nn.Linear(
            self.output_size[config.base_model], config.num_classes
        )
        self.resize = T.Resize((self.config.img_size,))

    def forward(self, batch: Batch) -> Tensor:
        """Forward Pass."""
        inputs = torch.concat((batch.h1_amplitude, batch.l1_amplitude), dim=1)

        # denoising frontend
        inputs = self.frontend(inputs)

        # prepare for efficientnet
        inputs = self.resize(inputs)

        # extract features from pretrained model
        x = self.encoder.extract_features(inputs)
        x = self.pool(x)
        x = self.out_proj(x)
        return x.to(torch.float32)
