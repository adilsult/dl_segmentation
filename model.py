from __future__ import annotations

# model.py — 2D U-Net model variants for brain tumor segmentation
#
# Three model architectures, each building on the previous:
#
#   UNet2D           — vanilla encoder-decoder with skip connections (baseline)
#   AttentionUNet2D  — same as UNet2D but with Attention Gates on skip connections
#   HybridUNet2D     — AttentionUNet2D + Transformer Bottleneck for global context
#
# All models:
#   - Accept input of shape (B, 4, 240, 240) — 4 MRI modalities as channels
#   - Output raw logits of shape (B, 1, 240, 240) — apply sigmoid externally
#   - Use base_channels=32 by default (or 16 for lighter CPU-friendly variants)
#
# Building blocks defined below (bottom-up):
#   DoubleConv -> Down -> Up -> AttentionGate -> AttentionUp -> TransformerBottleneck

import torch
import torch.nn as nn
import torch.nn.functional as F


# ===========================================================================
# Core Building Blocks
# ===========================================================================

class DoubleConv(nn.Module):
    """Two consecutive Conv2d -> BatchNorm -> ReLU blocks.

    This is the fundamental repeating unit in U-Net. Using two conv layers
    instead of one gives the network more depth per resolution level,
    allowing it to learn more complex features before down/upsampling.

    kernel_size=3, padding=1: preserves spatial dimensions (H, W stay the same).
    bias=False: not needed when BatchNorm follows (BN has its own bias parameter).
    inplace=True on ReLU: saves memory by modifying the tensor in place.
    """

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
    """Encoder downsampling block: MaxPool2d(2) -> DoubleConv.

    MaxPool halves spatial resolution (H, W -> H/2, W/2) while keeping channels.
    DoubleConv then typically doubles the channels.

    Channel progression example with base_channels=32:
      Down(32, 64):  output is 64 channels at half the spatial resolution
      Down(64, 128): output is 128 channels at quarter resolution
      ...
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(2),                         # stride=2: halves H and W
            DoubleConv(in_channels, out_channels),   # learn features at new resolution
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool_conv(x)


class Up(nn.Module):
    """Decoder upsampling block: ConvTranspose2d -> concat skip -> DoubleConv.

    ConvTranspose2d (learnable upsampling) doubles spatial resolution.
    The skip connection from the matching encoder level is concatenated
    to restore fine spatial details that were lost during downsampling.
    DoubleConv then refines the merged features.

    F.interpolate fallback: ConvTranspose2d can produce off-by-one sizes
    (e.g., 14 vs 15) due to rounding. We bilinearly resize to match the skip
    connection dimensions before concatenating.
    """

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        # ConvTranspose2d halves channels and doubles spatial resolution
        self.up   = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        # After concat: (in_channels//2 from decoder) + (skip_channels from encoder)
        self.conv = DoubleConv(in_channels // 2 + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Handle potential size mismatch from ConvTranspose2d (off-by-one in H or W)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        # Concatenate decoder features with encoder skip features along channel dim
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
    """Soft spatial attention gate applied to encoder skip connections.

    1. Project the gating signal g (from decoder, semantically rich) via W_g
    2. Project the skip features x (from encoder, spatially detailed) via W_x
    3. Both projections have the same spatial size after upsampling g
    4. Add them, apply ReLU, then collapse to a single-channel attention map via psi + sigmoid
    5. Multiply the attention map with the skip features: attended = x * alpha
       - alpha ≈ 1: keep this spatial region (likely tumor)
       - alpha ≈ 0: suppress this region (likely background)

    This is "soft" attention — values are continuous in (0, 1), not hard 0/1.
    The gate is learned end-to-end with the rest of the network via backprop.

    Args:
        g_channels:     channels in the gating signal (from decoder after upsample)
        x_channels:     channels in the skip connection (from encoder)
        inter_channels: intermediate dimension for the projection (usually = x_channels)
    """

    def __init__(self, g_channels: int, x_channels: int, inter_channels: int) -> None:
        super().__init__()
        # Project gate signal and skip features into shared intermediate space
        self.W_g = nn.Conv2d(g_channels, inter_channels, kernel_size=1, bias=True)
        self.W_x = nn.Conv2d(x_channels, inter_channels, kernel_size=1, bias=True)
        # Collapse to single-channel attention map via 1x1 conv + sigmoid
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),  # output in (0, 1) — continuous attention weight per pixel
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # Upsample gating signal to match skip spatial dimensions (g is lower-res)
        g_up  = F.interpolate(self.W_g(g), size=x.shape[-2:], mode="bilinear", align_corners=False)
        x_proj = self.W_x(x)
        # Additive attention: combine gate + skip, then collapse to attention map
        alpha = self.psi(self.relu(g_up + x_proj))  # shape: (B, 1, H, W)
        # Broadcast multiply: attends to relevant spatial positions in skip features
        return x * alpha


# ---------------------------------------------------------------------------
# AttentionUp: drop-in replacement for Up that adds an AttentionGate
# before the skip concatenation. Same __init__/forward interface as Up.
# ---------------------------------------------------------------------------
class AttentionUp(nn.Module):
    """Attention-gated decoder upsampling block.

    Same as Up but applies an AttentionGate to the skip connection
    before concatenating with the upsampled decoder features.

    The gating signal g = the decoder features after upsampling.
    The skip x = the encoder features at the same resolution level.
    """

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        # Attention gate: decoder upsampled features gate the skip connection
        self.attn_gate = AttentionGate(
            g_channels=in_channels // 2,   # channels after upsample (gating signal)
            x_channels=skip_channels,       # encoder skip channels (to be attended)
            inter_channels=skip_channels,   # intermediate dim = skip channels
        )
        self.conv = DoubleConv(in_channels // 2 + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        # Filter the skip features through the attention gate before concatenation
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
    """Vision Transformer bottleneck for capturing global spatial context.

    Placed between the CNN encoder (after 4 downsamples) and decoder.
    The bottleneck feature map is 15x15 with base_channels*16 channels.
    We treat each 15x15 spatial location as a "token" (like a word in NLP).

    Self-attention: each token attends to all 225 tokens, computing relationships
    between every pair of spatial locations. This gives the model a global
    "bird's eye view" of where the tumor is relative to all other brain structures —
    something purely local conv operations cannot achieve.

    Args:
        embed_dim:       channel dimension (= base_channels * 16)
        num_heads:       number of parallel attention heads (must divide embed_dim)
        num_layers:      number of stacked TransformerEncoder layers
        dim_feedforward: hidden size of the feedforward network within each layer
        dropout:         dropout rate for regularization
    """

    def __init__(self, embed_dim: int = 256, num_heads: int = 8,
                 num_layers: int = 4, dim_feedforward: int = 512,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        # Learnable positional embeddings: one vector per spatial token position (up to 225)
        # Initialized with small random values; learned during training
        self.pos_embed = nn.Parameter(torch.randn(1, 225, embed_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",   # GELU: smoother than ReLU, standard in modern transformers
            batch_first=True,    # expect input shape (batch, sequence, embed_dim)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)  # final normalization for training stability

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        # Flatten spatial dims into token sequence: (B, C, H, W) -> (B, H*W, C)
        # e.g., (B, 256, 15, 15) -> (B, 225, 256)
        tokens = x.flatten(2).transpose(1, 2)
        # Add positional encoding — transformer has no built-in position awareness
        tokens = tokens + self.pos_embed[:, :H * W, :]
        # Self-attention: every token attends to all 225 tokens (global context)
        tokens = self.transformer(tokens)
        tokens = self.norm(tokens)
        # Reshape back to spatial feature map: (B, H*W, C) -> (B, C, H, W)
        return tokens.transpose(1, 2).reshape(B, C, H, W)


# ===========================================================================
# Model Variants
# ===========================================================================

class UNet2D(nn.Module):
    """Standard 2D U-Net for binary brain tumor segmentation.

    Architecture:
        Encoder:    inc -> down1 -> down2 -> down3 -> bottleneck
        Channels:   4 -> 32 -> 64 -> 128 -> 256 -> 512 (with base_channels=32)
        Decoder:    up1 -> up2 -> up3 -> up4 -> outc
        Skip conns: encoder features copied to matching decoder levels

    Input:  (B, 4, 240, 240) — 4 MRI modalities
    Output: (B, 1, 240, 240) — raw logits (apply sigmoid for probabilities)

    At 240x240 input, spatial resolutions per level:
        inc:        240x240
        down1:      120x120
        down2:      60x60
        down3:      30x30
        bottleneck: 15x15
    """

    def __init__(self, in_channels: int = 4, out_channels: int = 1, base_channels: int = 32) -> None:
        super().__init__()
        # Encoder: progressively compress spatial dims, expand channel dims
        self.inc        = DoubleConv(in_channels, base_channels)            # 240x240
        self.down1      = Down(base_channels,      base_channels * 2)       # 120x120
        self.down2      = Down(base_channels * 2,  base_channels * 4)       # 60x60
        self.down3      = Down(base_channels * 4,  base_channels * 8)       # 30x30
        self.bottleneck = Down(base_channels * 8,  base_channels * 16)      # 15x15

        # Decoder: upsample back to original resolution using skip connections
        self.up1 = Up(base_channels * 16, base_channels * 8, base_channels * 8)
        self.up2 = Up(base_channels * 8,  base_channels * 4, base_channels * 4)
        self.up3 = Up(base_channels * 4,  base_channels * 2, base_channels * 2)
        self.up4 = Up(base_channels * 2,  base_channels,     base_channels)

        # Output head: 1x1 conv collapses all channels to 1 logit per pixel
        self.outc = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder forward pass — save each level's output for skip connections
        x1 = self.inc(x)         # 240x240
        x2 = self.down1(x1)      # 120x120
        x3 = self.down2(x2)      # 60x60
        x4 = self.down3(x3)      # 30x30
        x5 = self.bottleneck(x4) # 15x15

        # Decoder forward pass — each Up block receives current features + skip from encoder
        x = self.up1(x5, x4)    # 30x30  (skip from down3)
        x = self.up2(x, x3)     # 60x60  (skip from down2)
        x = self.up3(x, x2)     # 120x120 (skip from down1)
        x = self.up4(x, x1)     # 240x240 (skip from inc)
        return self.outc(x)     # (B, 1, 240, 240) — raw logits


# ---------------------------------------------------------------------------
# Attention U-Net: identical encoder to UNet2D, but every Up block is
# replaced with AttentionUp — adding learned spatial attention on each
# skip connection. This helps the model focus on tumor regions and
# suppress irrelevant background features passed through skip connections.
# Expected improvement: +2-4% Dice over vanilla U-Net on larger datasets.
# ---------------------------------------------------------------------------
class AttentionUNet2D(nn.Module):
    """U-Net with Attention Gates on all skip connections.

    Modification over UNet2D: every Up block replaced with AttentionUp.
    The encoder is identical to UNet2D — no changes to feature extraction.

    Attention Gates add ~11K parameters (498K vs 487K total for base_channels=32).
    Each skip connection is filtered by a learned spatial attention map before
    concatenation with the decoder features.

    On sufficient data, attention gates improve Dice by focusing gradient on
    tumor-relevant regions and suppressing background noise in skip connections.
    """

    def __init__(self, in_channels: int = 4, out_channels: int = 1, base_channels: int = 32) -> None:
        super().__init__()
        # Encoder — identical to UNet2D
        self.inc        = DoubleConv(in_channels, base_channels)
        self.down1      = Down(base_channels,     base_channels * 2)
        self.down2      = Down(base_channels * 2, base_channels * 4)
        self.down3      = Down(base_channels * 4, base_channels * 8)
        self.bottleneck = Down(base_channels * 8, base_channels * 16)

        # Decoder — AttentionUp instead of Up (attention gates on skip connections)
        self.up1 = AttentionUp(base_channels * 16, base_channels * 8, base_channels * 8)
        self.up2 = AttentionUp(base_channels * 8,  base_channels * 4, base_channels * 4)
        self.up3 = AttentionUp(base_channels * 4,  base_channels * 2, base_channels * 2)
        self.up4 = AttentionUp(base_channels * 2,  base_channels,     base_channels)

        self.outc = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder — same as UNet2D
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.bottleneck(x4)

        # Decoder — each AttentionUp filters skip connection before concatenation
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
    """Hybrid U-Net with Transformer Bottleneck + Attention Gates.

    Combines:
    - CNN encoder/decoder (same as UNet2D)
    - Attention gates on all skip connections (from AttentionUNet2D)
    - Transformer bottleneck between encoder and decoder (new addition)

    The transformer bottleneck processes the 15x15 bottleneck features
    as a sequence of 225 tokens with full self-attention — giving the
    model global spatial awareness before the decoder starts upsampling.

    With base_channels=32: ~2.8M parameters (vs 487K for UNet2D).
    The extra parameters come almost entirely from the transformer layers.

    Note: needs more data to outperform simpler models due to higher
    capacity. On small datasets (2 patients), UNet2D performs similarly.

    Args:
        in_channels:           number of input channels (4 MRI modalities)
        out_channels:          output channels (1 for binary segmentation)
        base_channels:         base channel width; all other widths are multiples of this
        num_heads:             attention heads in transformer (must divide base_channels*16)
        num_transformer_layers: number of stacked transformer encoder layers
        dim_feedforward:       feedforward hidden size in each transformer layer
        transformer_dropout:   dropout rate for transformer regularization
    """

    def __init__(self, in_channels: int = 4, out_channels: int = 1, base_channels: int = 32,
                 num_heads: int = 8, num_transformer_layers: int = 4,
                 dim_feedforward: int = 512, transformer_dropout: float = 0.1) -> None:
        super().__init__()
        # Encoder — same CNN backbone as UNet2D
        self.inc        = DoubleConv(in_channels, base_channels)
        self.down1      = Down(base_channels,     base_channels * 2)
        self.down2      = Down(base_channels * 2, base_channels * 4)
        self.down3      = Down(base_channels * 4, base_channels * 8)
        self.bottleneck = Down(base_channels * 8, base_channels * 16)

        # Transformer bottleneck — global self-attention at 15x15 spatial resolution
        # embed_dim = base_channels * 16 (matches bottleneck output channels)
        self.transformer = TransformerBottleneck(
            embed_dim=base_channels * 16,
            num_heads=num_heads,
            num_layers=num_transformer_layers,
            dim_feedforward=dim_feedforward,
            dropout=transformer_dropout,
        )

        # Decoder — AttentionUp for focused skip connections
        self.up1 = AttentionUp(base_channels * 16, base_channels * 8, base_channels * 8)
        self.up2 = AttentionUp(base_channels * 8,  base_channels * 4, base_channels * 4)
        self.up3 = AttentionUp(base_channels * 4,  base_channels * 2, base_channels * 2)
        self.up4 = AttentionUp(base_channels * 2,  base_channels,     base_channels)

        self.outc = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # CNN Encoder — compress image into deep feature representation
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.bottleneck(x4)  # shape: (B, base_channels*16, 15, 15)

        # Transformer bottleneck — add global context via self-attention at 15x15
        # Each of the 225 spatial positions attends to all other 225 positions
        x5 = self.transformer(x5)

        # Attention-gated decoder — upsample back to original resolution
        # Each level filters its skip connection through a learned attention gate
        x = self.up1(x5, x4)  # 30x30
        x = self.up2(x, x3)   # 60x60
        x = self.up3(x, x2)   # 120x120
        x = self.up4(x, x1)   # 240x240
        return self.outc(x)   # (B, 1, 240, 240) raw logits
