"""Layers."""
from torch import Tensor, nn


class Conv1dBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 16,
        stride: int = 1,
        padding: int = "same",
        dilation: int = 1,
        bias: bool = True,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.res_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.conv_1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        self.act_1 = nn.ReLU()
        self.norm_1 = nn.BatchNorm1d(out_channels)
        self.dropout_1 = nn.Dropout(dropout)

        self.conv_2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=kernel_size // 2,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        self.act_2 = nn.ReLU()
        self.norm_2 = nn.BatchNorm1d(out_channels)
        self.dropout_2 = nn.Dropout(dropout)

        self.conv_3 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=kernel_size // 2,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        self.act_3 = nn.ReLU()
        self.norm_3 = nn.BatchNorm1d(out_channels)
        self.dropout_3 = nn.Dropout(dropout)

        self.downsample = nn.MaxPool1d(kernel_size // 2, stride=2)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv_1(x)
        x = self.act_1(x)
        x = self.norm_1(x)
        x = self.dropout_1(x)

        x = self.conv_2(x)
        x = self.act_2(x)
        x = self.norm_2(x)
        x = self.dropout_2(x)

        x = self.conv_3(x)
        x = self.act_3(x)
        x = self.norm_3(x)
        x = self.dropout_3(x)

        x = self.downsample(x)
        return x
