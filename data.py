from __future__ import annotations

# data.py — BraTS data loading, normalization, and 2D slice extraction
#
# This module handles everything related to raw data:
# - Discovering BraTS case files (.tar archives or directories)
# - Extracting and loading the 4 MRI modalities (T1, T1ce, T2, FLAIR) from NIfTI files
# - Z-score normalization per modality per volume
# - Converting multi-class segmentation labels to a single binary tumor mask
# - Filtering and extracting 2D axial slices that contain tumor
#
# Output shape: X = (N, 4, 240, 240), y = (N, 1, 240, 240)
# where N = number of selected slices across all cases

import tarfile
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import nibabel as nib
import numpy as np


# BraTS 2021 uses different file naming conventions across cases/years.
# This lookup table maps each modality to all possible filename suffixes
# so the loader works regardless of the exact naming convention used.
MODALITY_ALIASES = {
    "t1":    ("_t1.nii.gz",   "_t1n.nii.gz",  "-t1.nii.gz",   "-t1n.nii.gz"),
    "t1ce":  ("_t1ce.nii.gz", "_t1c.nii.gz",  "-t1ce.nii.gz", "-t1c.nii.gz"),
    "t2":    ("_t2.nii.gz",   "_t2w.nii.gz",  "-t2.nii.gz",   "-t2w.nii.gz"),
    "flair": ("_flair.nii.gz","_t2f.nii.gz",  "-flair.nii.gz","-t2f.nii.gz"),
    "seg":   ("_seg.nii.gz",  "-seg.nii.gz"),
}


def extract_case_from_tar(
    tar_path: str | Path,
    extract_root: str | Path = ".cache/brats",
    force: bool = False,
) -> Path:
    """Extract a BraTS .tar archive to a local cache directory.

    Skips extraction if the target directory already contains .nii.gz files
    (unless force=True), so we don't re-extract on every run.
    """
    tar_path = Path(tar_path)
    extract_root = Path(extract_root)
    case_root = extract_root / tar_path.stem
    case_root.mkdir(parents=True, exist_ok=True)

    # Skip if already extracted — avoids re-doing slow I/O on every run
    existing_nii = list(case_root.glob("*.nii.gz"))
    if existing_nii and not force:
        return case_root

    with tarfile.open(tar_path, "r:*") as archive:
        archive.extractall(case_root)

    nested = list(case_root.glob("**/*.nii.gz"))
    if not nested:
        raise FileNotFoundError(f"No NIfTI files found after extracting {tar_path}")
    return case_root


def case_id_from_path(case_path: str | Path) -> str:
    """Extract a human-readable case ID from either a .tar path or a directory.

    Example: 'BraTS2021_00495.tar' -> 'BraTS2021_00495'
    """
    case_path = Path(case_path)
    if case_path.is_file():
        return case_path.stem   # filename without extension
    return case_path.name       # directory name


def _find_by_alias(files: Iterable[Path], aliases: Iterable[str]) -> Path:
    """Find the first file whose name matches any of the given suffix aliases.

    Used to locate each modality file regardless of naming convention.
    Raises FileNotFoundError if none of the aliases match.
    """
    lower_to_file = {f.name.lower(): f for f in files}
    for name, path in lower_to_file.items():
        for suffix in aliases:
            if name.endswith(suffix):
                return path
    raise FileNotFoundError(f"Could not find file for aliases: {aliases}")


def _dir_has_required_modalities(case_dir: Path) -> bool:
    """Check that a directory contains all 5 required BraTS files (4 modalities + seg mask).

    Used to distinguish real BraTS case directories from unrelated folders.
    """
    nii_files = [p for p in case_dir.glob("*.nii.gz")]
    if not nii_files:
        return False

    names = [p.name.lower() for p in nii_files]
    for aliases in MODALITY_ALIASES.values():
        if not any(any(name.endswith(suffix) for suffix in aliases) for name in names):
            return False
    return True


def discover_case_paths(input_path: str | Path, max_cases: int = 0) -> List[Path]:
    """Find all BraTS cases (as .tar files or case directories) under input_path.

    Supports three input types:
    - A single .tar file: returns just that file
    - A directory with .tar files: returns all .tar archives found recursively
    - A directory of extracted case dirs: returns all directories with required modalities

    max_cases: limit the number of returned cases (0 = no limit)
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    cases: List[Path] = []
    if input_path.is_file():
        # Single file provided directly
        cases = [input_path]
    else:
        # Check if the directory itself is a case (e.g., already extracted)
        if _dir_has_required_modalities(input_path):
            cases.append(input_path)

        # Find all .tar archives (skip hidden directories like .cache)
        tar_paths = sorted(
            p for p in input_path.rglob("*.tar") if not any(part.startswith(".") for part in p.parts)
        )
        cases.extend(tar_paths)

        # Find all subdirectories that look like BraTS cases
        case_dirs = sorted(
            p
            for p in input_path.rglob("*")
            if p.is_dir() and not any(part.startswith(".") for part in p.parts) and _dir_has_required_modalities(p)
        )
        cases.extend(case_dirs)

    # Deduplicate by case ID (e.g., avoid listing both the .tar and its extracted directory)
    unique: List[Path] = []
    seen_case_ids = set()
    for case in cases:
        cid = case_id_from_path(case).lower()
        if cid not in seen_case_ids:
            unique.append(case)
            seen_case_ids.add(cid)

    if max_cases and max_cases > 0:
        unique = unique[:max_cases]

    if not unique:
        raise FileNotFoundError(f"No BraTS cases found in: {input_path}")
    return unique


def resolve_case_dir(case_path: str | Path) -> Path:
    """Resolve a case path to an actual directory containing .nii.gz files.

    If the path is a .tar file, it extracts it first.
    If it's already a directory, it returns it directly.
    """
    case_path = Path(case_path)
    if case_path.is_file() and case_path.suffix == ".tar":
        return extract_case_from_tar(case_path)
    if case_path.is_dir():
        return case_path
    raise FileNotFoundError(f"Case path does not exist or unsupported type: {case_path}")


def load_case_volumes(case_path: str | Path) -> Dict[str, np.ndarray]:
    """Load all 5 NIfTI volumes for a BraTS case into memory as numpy arrays.

    Returns a dict with keys: 't1', 't1ce', 't2', 'flair', 'seg'
    Each array has shape (240, 240, 155) — the standard BraTS volume shape.

    Validates that all modalities have the same shape (sanity check).
    """
    case_dir = resolve_case_dir(case_path)
    nii_files = [p for p in case_dir.rglob("*.nii.gz")]
    if not nii_files:
        raise FileNotFoundError(f"No .nii.gz files found in {case_dir}")

    volumes: Dict[str, np.ndarray] = {}
    for key, aliases in MODALITY_ALIASES.items():
        file_path = _find_by_alias(nii_files, aliases)
        # nib.load reads the file; .get_fdata() returns the voxel data as float64
        # We cast to float32 to halve memory usage with no meaningful precision loss
        volumes[key] = nib.load(str(file_path)).get_fdata().astype(np.float32)

    # All modalities should have the same spatial shape — catch mismatches early
    shape = volumes["seg"].shape
    for key, volume in volumes.items():
        if volume.shape != shape:
            raise ValueError(f"Shape mismatch: {key} has {volume.shape}, expected {shape}")

    return volumes


def normalize_zscore(volume: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Apply z-score normalization: subtract mean, divide by std.

    Result: zero mean, unit variance (approximately).
    eps prevents division by zero for constant-valued regions (e.g., skull-stripped background).
    Applied per modality per volume — NOT globally across patients —
    because MRI intensities are not calibrated across scanners.
    """
    mean = float(np.mean(volume))
    std = float(np.std(volume))
    return ((volume - mean) / (std + eps)).astype(np.float32)


def normalize_modalities(
    volumes: Dict[str, np.ndarray],
    keys: Tuple[str, ...] = ("t1", "t1ce", "t2", "flair"),
) -> Dict[str, np.ndarray]:
    """Apply z-score normalization to each MRI modality independently.

    We normalize T1, T1ce, T2, FLAIR but NOT the segmentation mask
    (seg contains integer labels 0/1/2/4, not intensity values).
    """
    normalized = dict(volumes)
    for key in keys:
        normalized[key] = normalize_zscore(normalized[key])
    return normalized


def to_binary_mask(seg: np.ndarray) -> np.ndarray:
    """Convert BraTS multi-class segmentation labels to a single binary tumor mask.

    BraTS 2021 label convention:
      0 = background (healthy brain + non-brain)
      1 = necrotic tumor core (NCR)
      2 = peritumoral edema (ED)
      4 = enhancing tumor (ET)

    We merge all tumor classes (1, 2, 4) into a single binary mask:
      0 = no tumor, 1 = tumor (any subregion)

    This simplifies the problem to binary segmentation: tumor vs. background.
    """
    return (seg > 0).astype(np.float32)


def find_tumor_slices(mask_3d: np.ndarray, min_tumor_pixels: int = 1) -> List[int]:
    """Return the axial slice indices that contain at least min_tumor_pixels of tumor.

    BraTS volumes have 155 axial slices but many slices (especially at the top/bottom)
    contain no tumor at all. Training on blank slices would severely unbalance the
    dataset toward background. We filter them out here.

    mask_3d shape: (H, W, D) — D = number of axial slices (155 for BraTS)
    """
    slices: List[int] = []
    for idx in range(mask_3d.shape[-1]):
        if int(np.sum(mask_3d[:, :, idx])) >= min_tumor_pixels:
            slices.append(idx)
    return slices


def build_slices(
    volumes: Dict[str, np.ndarray],
    only_tumor: bool = True,
    min_tumor_pixels: int = 1,
    return_indices: bool = False,
) -> Tuple[np.ndarray, np.ndarray] | Tuple[np.ndarray, np.ndarray, List[int]]:
    """Convert 3D volumes into a stack of 2D axial slices.

    For each selected axial slice index, we:
    1. Stack the 4 modality slices into a (4, H, W) tensor — the 4-channel input
    2. Take the binary mask slice as (1, H, W) — the target

    Returns:
        X: (N, 4, 240, 240) float32 — input images
        y: (N, 1, 240, 240) float32 — binary tumor masks
        (optional) slice_indices: list of axial slice indices that were selected
    """
    seg = to_binary_mask(volumes["seg"])
    depth = seg.shape[-1]  # 155 axial slices in BraTS

    if only_tumor:
        # Keep only slices with at least 1 tumor pixel — avoids blank-slice imbalance
        slice_indices = find_tumor_slices(seg, min_tumor_pixels=min_tumor_pixels)
    else:
        slice_indices = list(range(depth))

    if not slice_indices:
        raise ValueError("No slices selected. Check mask or min_tumor_pixels.")

    x_slices = []
    y_slices = []
    for idx in slice_indices:
        # Stack the 4 modality channels for this axial slice: shape (4, 240, 240)
        image = np.stack(
            [
                volumes["t1"][:, :, idx],
                volumes["t1ce"][:, :, idx],
                volumes["t2"][:, :, idx],
                volumes["flair"][:, :, idx],
            ],
            axis=0,
        ).astype(np.float32)
        # Binary mask for this slice: shape (1, 240, 240) — channel dim needed by PyTorch
        mask = seg[:, :, idx][None, :, :].astype(np.float32)
        x_slices.append(image)
        y_slices.append(mask)

    # Stack all slices into batched arrays
    X = np.stack(x_slices, axis=0).astype(np.float32)   # (N, 4, 240, 240)
    y = np.stack(y_slices, axis=0).astype(np.float32)   # (N, 1, 240, 240)

    if return_indices:
        return X, y, slice_indices
    return X, y


def build_dataset_from_cases(
    case_paths: Sequence[str | Path],
    only_tumor: bool = True,
    min_tumor_pixels: int = 1,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, int | str]]]:
    """Load and concatenate 2D slice data from multiple BraTS cases.

    Processes each case independently (normalize per-case), then concatenates
    all slices into a single dataset. Returns a summary list for logging.

    Returns:
        X_cat: (N_total, 4, 240, 240) — all slices from all cases
        y_cat: (N_total, 1, 240, 240) — corresponding binary masks
        summary: list of dicts with case_id, num_slices, num_tumor_slices per case
    """
    x_all: List[np.ndarray] = []
    y_all: List[np.ndarray] = []
    summary: List[Dict[str, int | str]] = []

    for case in case_paths:
        case_id = case_id_from_path(case)
        # Load raw volumes and apply per-modality z-score normalization
        volumes = normalize_modalities(load_case_volumes(case))
        # Extract 2D slices; return_indices=True gives us the slice numbers for summary
        X, y, indices = build_slices(
            volumes,
            only_tumor=only_tumor,
            min_tumor_pixels=min_tumor_pixels,
            return_indices=True,
        )
        x_all.append(X)
        y_all.append(y)
        summary.append(
            {
                "case_id": case_id,
                "num_slices": int(X.shape[0]),
                "num_tumor_slices": int(len(indices)),
            }
        )

    if not x_all:
        raise ValueError("No slices were built from the provided case paths.")

    # Concatenate slices from all patients into one flat array
    X_cat = np.concatenate(x_all, axis=0).astype(np.float32)
    y_cat = np.concatenate(y_all, axis=0).astype(np.float32)
    return X_cat, y_cat, summary
