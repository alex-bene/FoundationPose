# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Pose refinement network that predicts rotation and translation corrections."""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn

from .network_modules import ConvBNReLU, PositionalEmbedding, ResnetBasicBlock


class RefineNet(nn.Module):
    """Network that refines pose estimates by predicting residual rotation and translation."""

    def __init__(
        self, use_batch_norm: bool, rotation_representation: Literal["axis_angle", "6d"], c_in: int = 4
    ) -> None:
        super().__init__()
        norm_layer = nn.BatchNorm2d if use_batch_norm else None

        self.encodeA = nn.Sequential(
            ConvBNReLU(C_in=c_in, C_out=64, kernel_size=7, stride=2, norm_layer=norm_layer),
            ConvBNReLU(C_in=64, C_out=128, kernel_size=3, stride=2, norm_layer=norm_layer),
            ResnetBasicBlock(128, 128, bias=True, norm_layer=norm_layer),
            ResnetBasicBlock(128, 128, bias=True, norm_layer=norm_layer),
        )

        self.encodeAB = nn.Sequential(
            ResnetBasicBlock(256, 256, bias=True, norm_layer=norm_layer),
            ResnetBasicBlock(256, 256, bias=True, norm_layer=norm_layer),
            ConvBNReLU(256, 512, kernel_size=3, stride=2, norm_layer=norm_layer),
            ResnetBasicBlock(512, 512, bias=True, norm_layer=norm_layer),
            ResnetBasicBlock(512, 512, bias=True, norm_layer=norm_layer),
        )

        embed_dim = 512
        num_heads = 4
        self.pos_embed = PositionalEmbedding(d_model=embed_dim, max_len=400)

        self.trans_head = nn.Sequential(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=512, batch_first=True),
            nn.Linear(512, 3),
        )

        if rotation_representation == "axis_angle":
            rot_out_dim = 3
        elif rotation_representation == "6d":
            rot_out_dim = 6
        else:
            raise RuntimeError
        self.rot_head = nn.Sequential(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=512, batch_first=True),
            nn.Linear(512, rot_out_dim),
        )

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> dict[str, torch.Tensor]:
        """Predict rotation and translation refinements from image pair (B, C, H, W)."""
        bs = len(A)
        output = {}

        x = torch.cat([A, B], dim=0)
        x = self.encodeA(x)
        a = x[:bs]
        b = x[bs:]

        ab = torch.cat((a, b), 1).contiguous()
        ab: torch.Tensor = self.encodeAB(ab)  # (B,C,H,W)

        ab = self.pos_embed(ab.reshape(bs, ab.shape[1], -1).permute(0, 2, 1))

        output["trans"] = self.trans_head(ab).mean(dim=1)
        output["rot"] = self.rot_head(ab).mean(dim=1)

        return output
