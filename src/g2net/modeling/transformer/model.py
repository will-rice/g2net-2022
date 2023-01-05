"""Huggingface Transformer Vision Model."""
import torch
from torch import Tensor, nn
from torchvision import transforms as VT
from transformers import AutoModel

from src.g2net import transforms as T
from src.g2net.data import Batch
from src.g2net.modeling.config import ModelConfig
from src.g2net.modeling.module import BaseLightningModule


class TransformerModel(BaseLightningModule):
    """Simple Model."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)

        self.encoder = AutoModel.from_pretrained(
            config.base_model,
            num_channels=config.in_channels,
            ignore_mismatched_sizes=True,
        )
        self.batch_norm = nn.BatchNorm1d(self.encoder.config.hidden_size)
        self.dropout = nn.Dropout(0.2)
        self.out_proj = nn.Linear(self.encoder.config.hidden_size, config.num_classes)

        self.prepare = nn.Sequential(
            VT.Resize((self.encoder.config.image_size, self.encoder.config.image_size)),
            VT.Normalize([0.5, 0.5], [0.5, 0.5]),
        )

        self.augment = VT.RandomApply(
            [
                T.VerticalShift(p=1.0),
                VT.RandomErasing(p=0.5),
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

        if self.training:
            with torch.no_grad():
                h1_amplitude = self.augment(h1_amplitude)
                l1_amplitude = self.augment(l1_amplitude)

        inputs = torch.concat((h1_amplitude, l1_amplitude), dim=1)
        inputs = self.prepare(inputs)

        x = self.encoder(inputs)[0]
        x = x[:, 0, :]
        x = self.batch_norm(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
