"""Render transformer_bottleneck.png — architecture diagram for HybridUNet2D."""
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

OUT = Path(__file__).parent / "transformer_bottleneck.png"

ENCODER_C = "#1A73E8"
BOTTLENECK_C = "#0F2A6E"
TRANSFORMER_C = "#C8162C"
DECODER_C = "#1E8E3E"
TOKEN_C = "#E37400"
TEXT_C = "#212121"
BG = "#FFFFFF"

fig, ax = plt.subplots(figsize=(16, 10), dpi=200)
ax.set_xlim(0, 16)
ax.set_ylim(0, 10)
ax.set_aspect("equal")
ax.axis("off")
fig.patch.set_facecolor(BG)


def block(x, y, w, h, color, label, sublabel=None, fontsize=11, edge="white"):
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle="round,pad=0.02,rounding_size=0.08",
                         linewidth=1.5, edgecolor=edge, facecolor=color)
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2 + (0.13 if sublabel else 0),
            label, ha="center", va="center",
            color="white", fontsize=fontsize, fontweight="bold")
    if sublabel:
        ax.text(x + w / 2, y + h / 2 - 0.20, sublabel,
                ha="center", va="center",
                color="white", fontsize=fontsize - 2, style="italic")


def arrow(x1, y1, x2, y2, color="#444", lw=1.8, style="->"):
    a = FancyArrowPatch((x1, y1), (x2, y2),
                        arrowstyle=style, mutation_scale=18,
                        color=color, linewidth=lw,
                        shrinkA=2, shrinkB=2)
    ax.add_patch(a)


# ---- Title ----
ax.text(8, 9.55, "HybridUNet2D — Transformer Bottleneck",
        ha="center", va="center", fontsize=20,
        fontweight="bold", color=BOTTLENECK_C)

# ============ ROW 1 (top): ENCODER → BOTTLENECK ============
# Encoder blocks
enc_specs = [
    ("Conv 16",  "240×240"),
    ("Conv 32",  "120×120"),
    ("Conv 64",  "60×60"),
    ("Conv 128", "30×30"),
]
ex0 = 0.4
ew, eh = 1.55, 0.95
egap = 0.35
ey = 7.6
positions_enc = []
for i, (lbl, sub) in enumerate(enc_specs):
    x = ex0 + i * (ew + egap)
    block(x, ey, ew, eh, ENCODER_C, lbl, sub, fontsize=11)
    positions_enc.append((x, ey, ew, eh))
for i in range(len(positions_enc) - 1):
    x1 = positions_enc[i][0] + ew
    y1 = ey + eh / 2
    x2 = positions_enc[i + 1][0]
    arrow(x1, y1, x2, y1)

ax.text((positions_enc[0][0] + positions_enc[-1][0] + ew) / 2, ey + eh + 0.4,
        "ENCODER", ha="center", fontsize=14,
        fontweight="bold", color=ENCODER_C)
ax.text((positions_enc[0][0] + positions_enc[-1][0] + ew) / 2, ey - 0.4,
        "↓ MaxPool 2×2 + DoubleConv at each level",
        ha="center", fontsize=10, color=TEXT_C, style="italic")

# Bottleneck on the right of encoder
bx = positions_enc[-1][0] + ew + 0.6
by, bw, bh = ey + 0.05, 1.4, 0.85
block(bx, by, bw, bh, BOTTLENECK_C, "256", "15 × 15", fontsize=12)
ax.text(bx + bw / 2, by + bh + 0.3, "bottleneck",
        ha="center", fontsize=11, color=BOTTLENECK_C, fontweight="bold")
ax.text(bx + bw / 2, by - 0.3, "(B, 256, 15, 15)",
        ha="center", fontsize=9, color=TEXT_C,
        style="italic", family="monospace")
arrow(positions_enc[-1][0] + ew, ey + eh / 2, bx, by + bh / 2)

# Down-arrow from bottleneck to transformer row
arrow(bx + bw / 2, by - 0.5, bx + bw / 2, 6.3,
      color=TOKEN_C, lw=2.2)
ax.text(bx + bw / 2 + 0.25, 6.85, "flatten",
        ha="left", fontsize=11, color=TOKEN_C, fontweight="bold")
ax.text(bx + bw / 2 + 0.25, 6.55, "15×15 → 225 tokens",
        ha="left", fontsize=9.5, color=TOKEN_C, style="italic")

# ============ ROW 2 (middle): TOKENS → TRANSFORMER → RESHAPE ============
# Token stack
tx, ty = 1.2, 4.95
tw, th = 0.22, 1.4
n_tokens_shown = 8
for i in range(n_tokens_shown):
    rect = Rectangle((tx + i * (tw + 0.05), ty), tw, th,
                     linewidth=1.0, edgecolor="white", facecolor=TOKEN_C)
    ax.add_patch(rect)
ax.text(tx + n_tokens_shown * (tw + 0.05) + 0.05, ty + th / 2,
        "…", fontsize=22, color=TOKEN_C, fontweight="bold",
        va="center")
for i in range(2):
    rect = Rectangle((tx + (n_tokens_shown + 1.5 + i) * (tw + 0.05), ty),
                     tw, th, linewidth=1.0, edgecolor="white",
                     facecolor=TOKEN_C)
    ax.add_patch(rect)
n_total = n_tokens_shown + 2
token_w = (n_total + 1.5) * (tw + 0.05)
token_cx = tx + token_w / 2
ax.text(token_cx, ty + th + 0.35,
        "225 tokens × dim 256", ha="center",
        fontsize=12, color=TOKEN_C, fontweight="bold")
ax.text(token_cx, ty - 0.35,
        "+ learnable positional embedding (1, 225, 256)",
        ha="center", fontsize=10, color=TEXT_C, style="italic")

# arrow from "flatten" point down to tokens
arrow(bx + bw / 2, 6.3, token_cx, ty + th + 0.05,
      color=TOKEN_C, lw=2.2)

# Arrow tokens → transformer
tend_x = tx + token_w + 0.05
arrow(tend_x, ty + th / 2, tend_x + 0.6, ty + th / 2,
      color=TOKEN_C, lw=2.2)

# Transformer × 2 (stacked depth illusion)
tr_x, tr_y, tr_w, tr_h = tend_x + 0.85, 4.7, 2.6, 1.9
# back layer (offset)
block(tr_x + 0.18, tr_y + 0.18, tr_w, tr_h, TRANSFORMER_C, "", fontsize=12)
# front layer
block(tr_x, tr_y, tr_w, tr_h, TRANSFORMER_C, "Transformer", "Encoder Layer",
      fontsize=14)
ax.text(tr_x + tr_w / 2, tr_y + tr_h + 0.35,
        "TransformerEncoder × 2",
        ha="center", fontsize=12, color=TRANSFORMER_C, fontweight="bold")
inside = (
    "MHSA (4 heads)  +  FFN (256 → 256, GELU)\n"
    "+ 2× LayerNorm  +  Dropout 0.1"
)
ax.text(tr_x + tr_w / 2, tr_y - 0.55, inside,
        ha="center", va="center", fontsize=10, color=TEXT_C,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#F5F5F5",
                  edgecolor="#BDBDBD", linewidth=0.8))

# Arrow transformer → reshape → bottleneck'
rs_x = tr_x + tr_w + 0.2
arrow(rs_x, tr_y + tr_h / 2, rs_x + 0.7, tr_y + tr_h / 2,
      color=TOKEN_C, lw=2.2)
ax.text(rs_x + 0.35, tr_y + tr_h / 2 + 0.22,
        "reshape", ha="center", fontsize=10,
        color=TOKEN_C, fontweight="bold")

# bottleneck' (post-transformer)
b2x, b2y, b2w, b2h = rs_x + 0.9, 5.2, 1.4, 0.9
block(b2x, b2y, b2w, b2h, BOTTLENECK_C, "256", "15 × 15", fontsize=12)
ax.text(b2x + b2w / 2, b2y - 0.3, "(B, 256, 15, 15)",
        ha="center", fontsize=9, color=TEXT_C,
        style="italic", family="monospace")

# Arrow from bottleneck' down to decoder row
arrow(b2x + b2w / 2, b2y - 0.5, b2x + b2w / 2, 3.1,
      color="#444", lw=1.8)

# ============ ROW 3 (bottom): DECODER ============
dec_specs = [
    ("AttUp 128", "30×30"),
    ("AttUp 64",  "60×60"),
    ("AttUp 32",  "120×120"),
    ("AttUp 16",  "240×240"),
]
# Right-to-left flow visually mirrors encoder
dec_w, dec_h = 1.55, 0.95
dec_gap = 0.35
# Decoder starts from right side and moves left? Better: keep left-to-right.
# Place decoder centered under transformer area
dx0 = b2x + b2w + 0.6  # not used; we'll reposition
# Total decoder width
total_dec_w = 4 * dec_w + 3 * dec_gap
dx0 = (16 - total_dec_w) / 2  # center
dy = 1.85
positions_dec = []
for i, (lbl, sub) in enumerate(dec_specs):
    x = dx0 + i * (dec_w + dec_gap)
    block(x, dy, dec_w, dec_h, DECODER_C, lbl, sub, fontsize=11)
    positions_dec.append((x, dy, dec_w, dec_h))
for i in range(len(positions_dec) - 1):
    x1 = positions_dec[i][0] + dec_w
    y1 = dy + dec_h / 2
    x2 = positions_dec[i + 1][0]
    arrow(x1, y1, x2, y1)

# arrow from bottleneck' down to first decoder block
arrow(b2x + b2w / 2, 3.1,
      positions_dec[0][0] + dec_w / 2, dy + dec_h)

ax.text((positions_dec[0][0] + positions_dec[-1][0] + dec_w) / 2,
        dy + dec_h + 0.4, "DECODER", ha="center", fontsize=14,
        fontweight="bold", color=DECODER_C)
ax.text((positions_dec[0][0] + positions_dec[-1][0] + dec_w) / 2,
        dy - 0.4,
        "↑ ConvTranspose 2×2  +  AttentionGate  +  DoubleConv",
        ha="center", fontsize=10, color=TEXT_C, style="italic")

# Output 1×1 conv after last decoder block
last = positions_dec[-1]
out_x = last[0] + dec_w + 0.4
arrow(last[0] + dec_w, dy + dec_h / 2, out_x, dy + dec_h / 2)
block(out_x, dy + 0.20, 0.6, 0.55, "#444444", "1×1", "Conv", fontsize=9)
ax.text(out_x + 0.3, dy - 0.15, "logits",
        ha="center", fontsize=9, color=TEXT_C, style="italic")

# ============ Bottom callout + footer ============
legend = (
    "Self-attention enables every spatial position at the bottleneck "
    "to attend to all 224 others —\nglobal context that pure convolutions cannot achieve."
)
ax.text(8, 0.85, legend, ha="center", va="center",
        fontsize=11, style="italic", color=BOTTLENECK_C,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#E8F0FE",
                  edgecolor=BOTTLENECK_C, linewidth=1.0))
ax.text(8, 0.05,
        "Total params: 2,836,629   |   +850K vs UNet2D   |   Hidden layers: 43",
        ha="center", fontsize=10.5, color=TEXT_C, fontweight="bold")

plt.savefig(OUT, dpi=200, bbox_inches="tight", pad_inches=0.25,
            facecolor=BG)
plt.close(fig)
print(f"Wrote {OUT}  ({OUT.stat().st_size/1024:.1f} KB)")
