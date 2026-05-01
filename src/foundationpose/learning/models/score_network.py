# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Scoring network that ranks pose hypotheses from multiple pairs."""

from __future__ import annotations

import torch
from torch import nn

from .network_modules import ConvBNReLU, PositionalEmbedding, ResnetBasicBlock


class ScoreNetMultiPair(nn.Module):
    """Network that scores and ranks multiple pose hypotheses via cross-attention."""

    def __init__(self, use_batch_norm: bool, c_in: int = 4) -> None:
        super().__init__()
        norm_layer = nn.BatchNorm2d if use_batch_norm else None

        self.encoderA = nn.Sequential(
            ConvBNReLU(C_in=c_in, C_out=64, kernel_size=7, stride=2, norm_layer=norm_layer),
            ConvBNReLU(C_in=64, C_out=128, kernel_size=3, stride=2, norm_layer=norm_layer),
            ResnetBasicBlock(128, 128, bias=True, norm_layer=norm_layer),
            ResnetBasicBlock(128, 128, bias=True, norm_layer=norm_layer),
        )

        self.encoderAB = nn.Sequential(
            ResnetBasicBlock(256, 256, bias=True, norm_layer=norm_layer),
            ResnetBasicBlock(256, 256, bias=True, norm_layer=norm_layer),
            ConvBNReLU(256, 512, kernel_size=3, stride=2, norm_layer=norm_layer),
            ResnetBasicBlock(512, 512, bias=True, norm_layer=norm_layer),
            ResnetBasicBlock(512, 512, bias=True, norm_layer=norm_layer),
        )

        embed_dim = 512
        num_heads = 4
        self.att = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, bias=True, batch_first=True)
        self.att_cross = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, bias=True, batch_first=True)

        self.pos_embed = PositionalEmbedding(d_model=embed_dim, max_len=400)
        self.linear = nn.Linear(embed_dim, 1)

    def extract_feat(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """Extract features from image pair A (B*L, C, H, W) and B."""
        bs = A.shape[0]  # B*L

        x = torch.cat([A, B], dim=0)
        x = self.encoderA(x)
        a = x[:bs]
        b = x[bs:]
        ab = torch.cat((a, b), dim=1)
        ab: torch.Tensor = self.encoderAB(ab)
        ab = self.pos_embed(ab.reshape(bs, ab.shape[1], -1).permute(0, 2, 1))
        ab, _ = self.att(ab, ab, ab)
        return ab.mean(dim=1).reshape(bs, -1)

    def forward(self, A: torch.Tensor, B: torch.Tensor, L: int) -> dict[str, torch.Tensor]:
        """Score pose hypotheses from A (B*L, C, H, W) with L pairs per batch element."""
        output = {}
        bs = A.shape[0] // L
        feats = self.extract_feat(A, B)  # (B*L, C)
        x = feats.reshape(bs, L, -1)
        x, _ = self.att_cross(x, x, x)

        output["score_logit"] = self.linear(x).reshape(bs, L)  # (B,L)

        return output
