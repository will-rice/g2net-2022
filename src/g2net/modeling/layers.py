"""General layers used by multiple models."""
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class GlobalAvgPool1d(nn.Module):
    """Global average pooling layer."""

    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return x.mean(dim=-1)


class GlobalAvgPool2d(nn.Module):
    """Global average pooling layer."""

    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return x.mean(dim=(2, 3))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, channels, seq_length]
        """
        x = x + self.pe[: x.size(2)]
        return self.dropout(x)


class LayerNorm(nn.Module):
    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        x = x.transpose(1, -1)
        x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)
        return x.transpose(1, -1)


class Subsample(nn.Module):
    """Spectrogram subsampling layer similar to what is used in ASR (Conformer, etc.)"""

    def __init__(self, in_channels, out_channels, kernel_size=31, stride: int = 4):
        super().__init__()
        self.conv_1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride
        )
        self.act_1 = nn.ReLU()
        self.conv_2 = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride
        )
        self.act_2 = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv_1(x)
        x = self.act_1(x)
        x = self.conv_2(x)
        x = self.act_2(x)
        return x
