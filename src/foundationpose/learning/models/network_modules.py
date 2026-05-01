# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Shared neural network building blocks for pose estimation models."""

from __future__ import annotations

import math
from typing import ClassVar

import torch
from torch import nn


class ConvBNReLU(nn.Module):
    """Conv-BatchNorm-ReLU block with configurable normalization."""

    def __init__(
        self,
        C_in: int,
        C_out: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        bias: bool = True,
        dilation: int = 1,
        norm_layer: type[nn.Module] | None = nn.BatchNorm2d,
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) // 2
        layers = [nn.Conv2d(C_in, C_out, kernel_size, stride, padding, groups=groups, bias=bias, dilation=dilation)]
        if norm_layer is not None:
            layers.append(norm_layer(C_out))
        layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the conv-bn-relu forward pass."""
        return self.net(x)


def conv3x3(
    in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1, bias: bool = False
) -> nn.Conv2d:
    """3x3 convolution with padding."""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=bias,
        dilation=dilation,
    )


class ResnetBasicBlock(nn.Module):
    """Basic residual block with two 3x3 convolutions."""

    __constants__: ClassVar[list[str]] = ["downsample"]

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: type[nn.Module] | None = nn.BatchNorm2d,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.norm_layer = norm_layer
        if groups != 1 or base_width != 64:
            msg = "BasicBlock only supports groups=1 and base_width=64"
            raise ValueError(msg)
        if dilation > 1:
            msg = "Dilation > 1 not supported in BasicBlock"
            raise NotImplementedError(msg)
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride, bias=bias)
        if self.norm_layer is not None:
            self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, bias=bias)
        if self.norm_layer is not None:
            self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the residual block forward pass."""
        identity = x

        out = self.conv1(x)
        if self.norm_layer is not None:
            out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if self.norm_layer is not None:
            out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class PositionalEmbedding(nn.Module):
    """Sinusoidal positional embedding for transformer inputs."""

    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.requires_grad = False  # og code had `require_grad` which is not a pytorch thing so did nothing

        position = torch.arange(0, max_len).float().unsqueeze(1)  # (N,1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()[None]

        pe[:, 0::2] = torch.sin(position * div_term)  # (N, d_model/2)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)  # (1, max_len, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input tensor of shape (B, N, D)."""
        return x + self.pe[:, : x.size(1)]
