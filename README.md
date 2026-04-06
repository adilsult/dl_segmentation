# MRI Brain Tumor Segmentation Using 2D U-Net

Binary brain tumor segmentation from multi-modal MRI scans using a 2D U-Net baseline trained on the [BraTS 2021](https://www.synapse.org/#!Synapse:syn27046444/wiki/616571) dataset.

## Overview

The pipeline takes 4 MRI modalities (T1, T1ce, T2, FLAIR) as input and outputs a binary tumor segmentation mask. Each 3D volume is processed as 2D axial slices, producing input tensors of shape `(N, 4, 240, 240)`.

**Current results (2-patient baseline):**

| Metric | Value |
|--------|-------|
| Val Dice | 0.9546 |
| Val IoU | 0.9132 |
| Train Loss | 0.75 → 0.11 |

## Project Structure

```
├── data.py          # Data loading, normalization, slice extraction
├── dataset.py       # PyTorch Dataset and DataLoader utilities
├── model.py         # 2D U-Net architecture
├── losses.py        # Dice + BCE combined loss
├── eval.py          # Evaluation metrics (Dice, IoU, Precision, Recall)
├── train.py         # Training loop with early stopping
├── DLmidtermproject.ipynb  # Main notebook (works locally and in Google Colab)
└── runs/            # Training artifacts (history.json files)
```

## Quick Start

### Google Colab (recommended)

Open `DLmidtermproject.ipynb` in Google Colab and press **Run All**. The notebook automatically downloads all required files (dataset, checkpoints) via `gdown` — no setup needed.

### Local Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch nibabel numpy matplotlib
```

Place BraTS `.tar` case files in the project directory, then run:

```bash
# Train on a single case
python train.py --epochs 20 --batch-size 4 --cpu \
  --case-path BraTS2021_00495.tar \
  --early-stopping-patience 5 \
  --output-dir runs/baseline

# Train on multiple cases (place .tar files in a folder)
python train.py --epochs 30 --batch-size 4 --cpu \
  --case-path /path/to/cases/ \
  --val-ratio 0.2 \
  --early-stopping-patience 8 \
  --output-dir runs/multicase
```

## Model Architecture

2D U-Net with encoder-decoder structure and skip connections:

- **Input:** 4 channels (T1, T1ce, T2, FLAIR), 240x240
- **Encoder:** 4 downsampling blocks (Conv3x3 + BN + ReLU + MaxPool)
- **Bottleneck:** 256 channels
- **Decoder:** 4 upsampling blocks (ConvTranspose + skip connection + Conv3x3)
- **Output:** 1 channel, sigmoid activation, binary mask

## Loss Function

Combined Dice Loss + Binary Cross-Entropy with equal weights (0.5 / 0.5), providing both region-based and pixel-level gradient signals.

## Team

Adilsultan Khairolla, Kavyashree Markuli Vijaykumar, Reda

AI700-001_BK — Spring 2026
