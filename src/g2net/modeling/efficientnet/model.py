"""Model"""
from typing import Any, Optional

import timm
import torch
import torchvision.transforms as VT
from torch import Tensor, nn

from src.g2net import transforms as T
from src.g2net.data import Batch
from src.g2net.modeling.config import ModelConfig
from src.g2net.modeling.layers import GlobalAvgPool2d, Subsample
from src.g2net.modeling.module import BaseLightningModule


class EfficientNetModel(BaseLightningModule):
    """Simple Model."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)

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
        self.img_size = {
            "efficientnet-b0": 224,
            "efficientnet-b1": 240,
            "efficientnet-b2": 260,
            "efficientnet-b3": 300,
            "efficientnet-b4": 380,
            "efficientnet-b5": 456,
            "efficientnet-b6": 528,
            "efficientnet-b7": 600,
        }
        self.encoder = timm.create_model(
            self.config.base_model, pretrained=True, in_chans=self.config.in_channels
        )
        # surely there is a better way to do this part
        clsf = self.encoder.default_cfg["classifier"]
        n_features = self.encoder._modules[clsf].in_features
        self.encoder._modules[clsf] = nn.Identity()

        self.dropout = nn.Dropout(config.dropout)
        self.out_proj = nn.Linear(n_features, config.num_classes)
        self.resize = VT.Resize((self.config.img_size,))

        self.augment = VT.RandomApply(
            [
                T.VerticalShift(p=1.0),
                VT.RandomHorizontalFlip(p=0.5),
                VT.RandomVerticalFlip(p=0.5),
                T.TimeMasking(10, num_masks=1),
                T.FreqMasking(10, num_masks=2),
            ],
            p=config.augment_p,
        )

    def forward(self, batch: Batch) -> Tensor:
        """Forward Pass."""
        h1_amplitude = batch.h1_amplitude
        l1_amplitude = batch.l1_amplitude

        with torch.no_grad():
            h1_amplitude = self.augment(h1_amplitude)
            l1_amplitude = self.augment(l1_amplitude)

        inputs = torch.concat((h1_amplitude, l1_amplitude), dim=1)
        inputs = self.resize(inputs)

        # extract features from pretrained model
        x = self.encoder(inputs)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x.to(torch.float32)

    def freeze_encoder(self) -> None:
        """Freeze pretrained encoder."""
        for param in self.encoder.parameters():
            param.requires_grad = False
