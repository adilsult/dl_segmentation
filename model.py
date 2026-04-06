from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.pool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels // 2 + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


# ---------------------------------------------------------------------------
# Attention Gate (Oktay et al., 2018)
# Learns a spatial attention map that highlights relevant regions in the
# encoder skip features before they are concatenated with decoder features.
# g = gating signal from decoder (lower-res, semantically rich)
# x = skip connection from encoder (higher-res, spatially detailed)
# The gate suppresses irrelevant background features in the skip path.
# ---------------------------------------------------------------------------
class AttentionGate(nn.Module):
    def __init__(self, g_channels: int, x_channels: int, inter_channels: int) -> None:
        super().__init__()
        # Project gate signal and skip features into shared intermediate space
        self.W_g = nn.Conv2d(g_channels, inter_channels, kernel_size=1, bias=True)
        self.W_x = nn.Conv2d(x_channels, inter_channels, kernel_size=1, bias=True)
        # Collapse to single-channel attention map via 1x1 conv + sigmoid
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # Upsample gating signal to match skip spatial dimensions
        g_up = F.interpolate(self.W_g(g), size=x.shape[-2:], mode="bilinear", align_corners=False)
        x_proj = self.W_x(x)
        # Additive attention: combine gate + skip, then collapse to attention map
        alpha = self.psi(self.relu(g_up + x_proj))
        # Element-wise multiply: zero out irrelevant skip regions
        return x * alpha


# ---------------------------------------------------------------------------
# AttentionUp: drop-in replacement for Up that adds an AttentionGate
# before the skip concatenation. Same __init__/forward interface as Up.
# ---------------------------------------------------------------------------
class AttentionUp(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        # Attention gate: decoder upsampled features gate the skip connection
        self.attn_gate = AttentionGate(
            g_channels=in_channels // 2,   # channels after upsample
            x_channels=skip_channels,       # encoder skip channels
            inter_channels=skip_channels,   # intermediate dim = skip channels
        )
        self.conv = DoubleConv(in_channels // 2 + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        # Apply attention gate to skip connection before concatenation
        skip = self.attn_gate(g=x, x=skip)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


# ---------------------------------------------------------------------------
# Transformer Bottleneck (inspired by TransBTS, Wang et al., 2021)
# Inserted between the CNN encoder and decoder to capture long-range
# spatial dependencies that convolutions miss (limited receptive field).
#
# How it works:
# 1. Flatten bottleneck feature map (B, C, H, W) -> (B, H*W, C) tokens
# 2. Add learnable positional encoding so the transformer knows spatial layout
# 3. Process through multi-head self-attention layers (each token attends
#    to ALL other tokens — global context across the entire image)
# 4. Reshape back to (B, C, H, W) for the decoder
#
# With 240x240 input and 4 downsamples, bottleneck is 15x15 = 225 tokens,
# which is computationally lightweight for self-attention.
# ---------------------------------------------------------------------------
class TransformerBottleneck(nn.Module):
    def __init__(self, embed_dim: int = 256, num_heads: int = 8,
                 num_layers: int = 4, dim_feedforward: int = 512,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        # Learnable position embedding for 15x15 = 225 spatial tokens
        self.pos_embed = nn.Parameter(torch.randn(1, 225, embed_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",       # GELU activation (standard in ViT)
            batch_first=True,        # input shape: (batch, seq, dim)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)  # stabilize output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        # Flatten spatial dims into token sequence: (B, C, H, W) -> (B, H*W, C)
        tokens = x.flatten(2).transpose(1, 2)
        # Add positional encoding so transformer knows spatial layout
        tokens = tokens + self.pos_embed[:, :H * W, :]
        # Self-attention: each token attends to all 225 tokens (global context)
        tokens = self.transformer(tokens)
        tokens = self.norm(tokens)
        # Reshape back to spatial feature map: (B, H*W, C) -> (B, C, H, W)
        return tokens.transpose(1, 2).reshape(B, C, H, W)


# ===========================================================================
# Model Variants
# ===========================================================================

class UNet2D(nn.Module):
    def __init__(self, in_channels: int = 4, out_channels: int = 1, base_channels: int = 32) -> None:
        super().__init__()
        self.inc = DoubleConv(in_channels, base_channels)
        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8)
        self.bottleneck = Down(base_channels * 8, base_channels * 16)

        self.up1 = Up(base_channels * 16, base_channels * 8, base_channels * 8)
        self.up2 = Up(base_channels * 8, base_channels * 4, base_channels * 4)
        self.up3 = Up(base_channels * 4, base_channels * 2, base_channels * 2)
        self.up4 = Up(base_channels * 2, base_channels, base_channels)

        self.outc = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.bottleneck(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)


# ---------------------------------------------------------------------------
# Attention U-Net: identical encoder to UNet2D, but every Up block is
# replaced with AttentionUp — adding learned spatial attention on each
# skip connection. This helps the model focus on tumor regions and
# suppress irrelevant background features passed through skip connections.
# Expected improvement: +2-4% Dice over vanilla U-Net.
# ---------------------------------------------------------------------------
class AttentionUNet2D(nn.Module):
    def __init__(self, in_channels: int = 4, out_channels: int = 1, base_channels: int = 32) -> None:
        super().__init__()
        # Encoder — same as UNet2D
        self.inc = DoubleConv(in_channels, base_channels)
        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8)
        self.bottleneck = Down(base_channels * 8, base_channels * 16)

        # Decoder — AttentionUp instead of Up (attention gates on skip connections)
        self.up1 = AttentionUp(base_channels * 16, base_channels * 8, base_channels * 8)
        self.up2 = AttentionUp(base_channels * 8, base_channels * 4, base_channels * 4)
        self.up3 = AttentionUp(base_channels * 4, base_channels * 2, base_channels * 2)
        self.up4 = AttentionUp(base_channels * 2, base_channels, base_channels)

        self.outc = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.bottleneck(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)


# ---------------------------------------------------------------------------
# Hybrid U-Net: combines BOTH improvements:
# 1. Transformer at the bottleneck — captures global spatial context
#    (each pixel attends to every other pixel across the entire image)
# 2. Attention gates on skip connections — focuses on tumor regions
#
# This is the most powerful variant: the transformer provides the "big
# picture" understanding (where the tumor is relative to brain structures)
# while attention gates refine the spatial details at each resolution level.
#
# Architecture flow:
#   Encoder -> Bottleneck -> [Transformer] -> Attention Decoder
# ---------------------------------------------------------------------------
class HybridUNet2D(nn.Module):
    def __init__(self, in_channels: int = 4, out_channels: int = 1, base_channels: int = 32,
                 num_heads: int = 8, num_transformer_layers: int = 4,
                 dim_feedforward: int = 512, transformer_dropout: float = 0.1) -> None:
        super().__init__()
        # Encoder — same CNN backbone as UNet2D
        self.inc = DoubleConv(in_channels, base_channels)
        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8)
        self.bottleneck = Down(base_channels * 8, base_channels * 16)

        # Transformer bottleneck — global context via self-attention
        self.transformer = TransformerBottleneck(
            embed_dim=base_channels * 16,
            num_heads=num_heads,
            num_layers=num_transformer_layers,
            dim_feedforward=dim_feedforward,
            dropout=transformer_dropout,
        )

        # Decoder — AttentionUp for focused skip connections
        self.up1 = AttentionUp(base_channels * 16, base_channels * 8, base_channels * 8)
        self.up2 = AttentionUp(base_channels * 8, base_channels * 4, base_channels * 4)
        self.up3 = AttentionUp(base_channels * 4, base_channels * 2, base_channels * 2)
        self.up4 = AttentionUp(base_channels * 2, base_channels, base_channels)

        self.outc = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.bottleneck(x4)

        # Transformer: add global context at the bottleneck
        x5 = self.transformer(x5)

        # Decoder with attention-gated skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)

