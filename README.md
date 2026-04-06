# MRI Brain Tumor Segmentation Using 2D U-Net

Binary brain tumor segmentation from multi-modal MRI scans using 2D U-Net variants trained on the [BraTS 2021](https://www.synapse.org/#!Synapse:syn27046444/wiki/616571) dataset.

## Overview

The pipeline takes 4 MRI modalities (T1, T1ce, T2, FLAIR) as input and outputs a binary tumor segmentation mask. Each 3D volume is processed as 2D axial slices, producing input tensors of shape `(N, 4, 240, 240)`.

**Model comparison (2 patients, 133 slices, 30 epochs):**

| Model | Loss | Params | Best Dice | Best IoU |
|-------|------|--------|-----------|----------|
| UNet2D | DiceBCE | 486,913 | **0.9608** | **0.9246** |
| AttentionUNet2D | DiceBCE | 498,341 | 0.9584 | 0.9201 |
| HybridUNet2D | DiceBCE | 2,836,629 | 0.9574 | 0.9183 |
| HybridUNet2D | FocalTversky | 2,836,629 | 0.9167 | 0.8462 |

## Prerequisites

Install these before starting (skip any you already have):

| Tool | How to install |
|------|---------------|
| **Python 3.9+** | [python.org/downloads](https://www.python.org/downloads/) — check "Add Python to PATH" during install |
| **Git** | [git-scm.com/downloads](https://git-scm.com/downloads) — needed to clone the repo |
| **Jupyter** *(optional)* | Installed automatically below, or use VS Code with the Python extension |

To verify they are installed, open a terminal and run:
```
python --version
git --version
```

## Quick Start

### Option 1: macOS / Linux (Terminal)

```bash
git clone https://github.com/adilsult/dl_segmentation.git
cd dl_segmentation
python3 -m venv .venv
source .venv/bin/activate
pip install torch nibabel numpy matplotlib scikit-learn jupyter
jupyter notebook DLmidtermproject.ipynb
```

### Option 2: Windows (Command Prompt)

```cmd
git clone https://github.com/adilsult/dl_segmentation.git
cd dl_segmentation
python -m venv .venv
.venv\Scripts\activate
pip install torch nibabel numpy matplotlib scikit-learn jupyter
jupyter notebook DLmidtermproject.ipynb
```

### Option 3: Windows (PowerShell)

```powershell
git clone https://github.com/adilsult/dl_segmentation.git
cd dl_segmentation
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install torch nibabel numpy matplotlib scikit-learn jupyter
jupyter notebook DLmidtermproject.ipynb
```

> **Note:** On Windows PowerShell, if you get an execution policy error, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` first.

### Option 4: No install — download ZIP

If you don't have Git, go to [github.com/adilsult/dl_segmentation](https://github.com/adilsult/dl_segmentation), click the green **Code** button → **Download ZIP**, extract it, then follow the steps above starting from `cd dl_segmentation`.

### Run the notebook

Open `DLmidtermproject.ipynb` in Jupyter or VS Code and press **Run All**.

```bash
pip install jupyter
jupyter notebook DLmidtermproject.ipynb
```

All datasets, checkpoints, and training histories are included in the repository — no external downloads needed.

## Project Structure

```
├── data.py                   # Data loading, normalization, slice extraction
├── dataset.py                # PyTorch Dataset and DataLoader utilities
├── model.py                  # UNet2D, AttentionUNet2D, HybridUNet2D
├── losses.py                 # DiceBCELoss, FocalTverskyLoss
├── eval.py                   # Evaluation metrics (Dice, IoU, Precision, Recall)
├── train.py                  # CLI training loop with early stopping
├── compare_models.py         # Multi-model comparison script
├── BraTS2021_00495.tar       # BraTS case 1 (~10 MB)
├── BraTS2021_00621.tar       # BraTS case 2 (~10 MB)
├── DLmidtermproject.ipynb    # Main notebook — Run All to reproduce results
└── runs/                     # Pre-trained results
    ├── baseline_2d_e20/      # 20-epoch UNet2D baseline (history + checkpoint)
    ├── unet_dicebce/         # UNet2D + DiceBCE (history + checkpoint)
    ├── attn_dicebce/         # AttentionUNet2D + DiceBCE (history + checkpoint)
    ├── hybrid_dicebce/       # HybridUNet2D + DiceBCE (history only)
    └── hybrid_focal_tversky/ # HybridUNet2D + FocalTversky (history only)
```

## Model Architectures

- **UNet2D**: Standard encoder-decoder with skip connections (4 down/up blocks, base 16 channels)
- **AttentionUNet2D**: Adds learned attention gates on each skip connection to focus on tumor regions
- **HybridUNet2D**: Combines a Transformer bottleneck (2-layer, 4-head self-attention at 15x15 resolution) with attention-gated skip connections

## Loss Functions

- **DiceBCELoss**: 0.5 × Dice Loss + 0.5 × BCE — combines region-based and pixel-level supervision
- **FocalTverskyLoss**: Tversky Index with focal modulation (alpha=0.7, beta=0.3, gamma=0.75) — penalizes missed tumor pixels more heavily

## CLI Training

```bash
# Single model
python train.py --epochs 30 --batch-size 4 --cpu \
  --model-type attention_unet --loss dicebce \
  --case-path BraTS2021_00495.tar \
  --output-dir runs/my_run

# All 4 configurations
python compare_models.py
```

## Team

Adilsultan Khairolla, Kavyashree M.V.
AI700-001_BK — Spring 2026
