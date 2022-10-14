"""Audio augmentation transforms"""
import random
from typing import Optional, Union

import numpy as np
import torch
import torchaudio.transforms as AT
from torch import Tensor, nn


class GaussianNoise(nn.Module):
    """Gaussian Noise Transform."""

    def __init__(
        self,
        min_intensity: float = 0.0,
        max_intensity: float = 10.0,
        probability: float = 0.5,
    ):
        super().__init__()
        self.intensity_dist = torch.distributions.uniform.Uniform(
            min_intensity, max_intensity
        )
        self.probability = probability

    def forward(self, x: Union[Tensor, np.ndarray]) -> Tensor:
        """Forward Pass."""
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)

        if random.random() < self.probability:

            intensity = self.intensity_dist.sample()
            noise = torch.randn_like(x) * intensity
            x += noise

        return x


class TimeMasking(nn.Module):
    def __init__(self, size=10, num_masks=1, p=0.5):
        super().__init__()
        self.num_masks = num_masks
        self.transform = AT.TimeMasking(size, p=p)

    def forward(self, x: Tensor) -> Tensor:
        for _ in range(self.num_masks):
            x = self.transform(x)
        return x


class FreqMasking(nn.Module):
    def __init__(self, size=10, num_masks=3, p=0.5):
        super().__init__()
        self.size = size
        self.num_masks = num_masks
        self.p = p
        self.transform = AT.FrequencyMasking(size)

    def forward(self, x: Tensor) -> Tensor:
        if random.random() < self.p:
            for _ in range(self.num_masks):
                x = self.transform(x)
        return x


class NoiseTimeMasking(nn.Module):
    def __init__(self, size, p=1.0):
        super().__init__()
        self.size = size
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        x = noise_mask_along_axis(x, mask_param=self.size, axis=1, p=self.p)
        return x


class NoiseFrequencyMasking(nn.Module):
    def __init__(self, size, p=1.0):
        super().__init__()
        self.size = size
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        x = noise_mask_along_axis(x, mask_param=self.size, axis=2, p=self.p)
        return x


class VerticalShift(nn.Module):
    def __init__(self, p=1.0):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:

        if random.random() < self.p:
            x = torch.roll(x, shifts=np.random.randint(low=0, high=x.shape[1]), dims=1)
        return x


class CutOut(nn.Module):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """

    def __init__(self, n_holes, length):
        super().__init__()
        self.n_holes = n_holes
        self.length = length

    def forward(self, img: Tensor) -> Tensor:
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = torch.ones((h, w), dtype=torch.float32)

        for n in range(self.n_holes):
            y = torch.randint(size=(), high=h)
            x = torch.randint(size=(), high=w)

            y1 = torch.clamp(y - self.length // 2, 0, h)
            y2 = torch.clamp(y + self.length // 2, 0, h)
            x1 = torch.clamp(x - self.length // 2, 0, w)
            x2 = torch.clamp(x + self.length // 2, 0, w)

            mask[y1:y2, x1:x2] = 0.0

        mask = mask.expand_as(img)
        img = img * mask.to(img.device)

        return img


class NoiseOut(nn.Module):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
        intensity (float): The intensity of the noise to be added.
    """

    def __init__(self, n_holes: int, length: int, intensity: Optional[float] = None):
        super().__init__()
        self.n_holes = n_holes
        self.length = length
        self.intensity = intensity

    def forward(self, img: Tensor) -> Tensor:
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = torch.ones((h, w), dtype=torch.float32)

        for n in range(self.n_holes):
            y = torch.randint(size=(), high=h)
            x = torch.randint(size=(), high=w)

            y1 = torch.clamp(y - self.length // 2, 0, h)
            y2 = torch.clamp(y + self.length // 2, 0, h)
            x1 = torch.clamp(x - self.length // 2, 0, w)
            x2 = torch.clamp(x + self.length // 2, 0, w)

            noise = (
                torch.randn_like(mask[y1:y2, x1:x2]) * self.intensity
                if self.intensity
                else random.random()
            )
            mask[y1:y2, x1:x2] = noise

        mask = mask.expand_as(img)
        img *= mask.to(img.device)

        return img


def _get_mask_param(mask_param: int, p: float, axis_length: int) -> int:
    if p == 1.0:
        return mask_param
    else:
        return min(mask_param, int(axis_length * p))


def noise_mask_along_axis(
    specgram: Tensor,
    mask_param: int,
    axis: int,
    p: float = 1.0,
) -> Tensor:
    r"""Apply a mask along ``axis``.
    .. devices:: CPU CUDA
    .. properties:: Autograd TorchScript
    Mask will be applied from indices ``[v_0, v_0 + v)``,
    where ``v`` is sampled from ``uniform(0, max_v)`` and
    ``v_0`` from ``uniform(0, specgrams.size(axis) - v)``, with
    ``max_v = mask_param`` when ``p = 1.0`` and
    ``max_v = min(mask_param, floor(specgrams.size(axis) * p))``
    otherwise.
    All examples will have the same mask interval.
    Args:
        specgram (Tensor): Real spectrogram `(channel, freq, time)`
        mask_param (int): Number of columns to be masked will be uniformly
        sampled from [0, mask_param]
        mask_value (float): Value to assign to the masked columns
        axis (int): Axis to apply masking on (1 -> frequency, 2 -> time)
        p (float, optional): maximum proportion of columns
        that can be masked. (Default: 1.0)
    Returns:
        Tensor: Masked spectrogram of dimensions `(channel, freq, time)`
    """
    if axis not in [1, 2]:
        raise ValueError("Only Frequency and Time masking are supported")

    if not 0.0 <= p <= 1.0:
        raise ValueError(f"The value of p must be between 0.0 and 1.0 ({p} given).")

    mask_param = _get_mask_param(mask_param, p, specgram.shape[axis])
    if mask_param < 1:
        return specgram

    # pack batch
    shape = specgram.size()
    specgram = specgram.reshape([-1] + list(shape[-2:]))
    value = torch.rand(1) * mask_param
    min_value = torch.rand(1) * (specgram.size(axis) - value)

    mask_start = (min_value.long()).squeeze()
    mask_end = (min_value.long() + value.long()).squeeze()
    mask = torch.arange(
        0, specgram.shape[axis], device=specgram.device, dtype=specgram.dtype
    )
    mask = (mask >= mask_start) & (mask < mask_end)
    if axis == 1:
        mask = mask.unsqueeze(-1)

    if mask_end - mask_start >= mask_param:
        raise ValueError(
            "Number of columns to be masked should be less than mask_param"
        )

    noise = torch.randn_like(specgram) * mask

    specgram = specgram + noise

    # unpack batch
    specgram = specgram.reshape(shape[:-2] + specgram.shape[-2:])

    return specgram
