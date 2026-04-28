from __future__ import annotations

# data.py — BraTS data loading, normalization, and 2D slice extraction
#
# Supports two dataset formats:
#
#   BraTS 2021 (.tar, one case per archive):
#     BraTS2021_00495/
#       ├── *_t1.nii.gz
#       ├── *_t1ce.nii.gz
#       ├── *_t2.nii.gz
#       ├── *_flair.nii.gz
#       └── *_seg.nii.gz          ← mask co-located with modalities
#
#   BraTS 2024 GLI (.tar.gz, multiple cases per archive):
#     data/
#       └── BraTS-GLI-00001-000/
#             ├── *-t1c.nii.gz
#             ├── *-t1n.nii.gz
#             ├── *-t2w.nii.gz
#             └── *-t2f.nii.gz
#     labels/
#       └── BraTS-GLI-00001-000-seg.nii.gz   ← mask in separate folder
#
# The BraTS 2024 format is normalised on extraction: the seg file is copied
# into each case directory so all downstream code stays format-agnostic.
#
# Output shape: X = (N, 4, 240, 240), y = (N, 1, 240, 240)

import shutil
import tarfile
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import nibabel as nib
import numpy as np


# BraTS naming varies across years. This table maps each modality to all
# known suffixes so the loader works with both BraTS 2021 and 2024.
MODALITY_ALIASES = {
    "t1":    ("_t1.nii.gz",    "_t1n.nii.gz",  "-t1.nii.gz",   "-t1n.nii.gz"),
    "t1ce":  ("_t1ce.nii.gz",  "_t1c.nii.gz",  "-t1ce.nii.gz", "-t1c.nii.gz"),
    "t2":    ("_t2.nii.gz",    "_t2w.nii.gz",  "-t2.nii.gz",   "-t2w.nii.gz"),
    "flair": ("_flair.nii.gz", "_t2f.nii.gz",  "-flair.nii.gz","-t2f.nii.gz"),
    "seg":   ("_seg.nii.gz",   "-seg.nii.gz"),
}


# ---------------------------------------------------------------------------
# Archive extraction helpers
# ---------------------------------------------------------------------------

def _is_brats2024_root(path: Path) -> bool:
    """Return True if path is a BraTS 2024 GLI root (has data/ and labels/ subdirs)."""
    return (path / "data").is_dir() and (path / "labels").is_dir()


def _normalize_brats2024_cases(root: Path, case_extract_root: Path) -> List[Path]:
    """Normalise a BraTS 2024 GLI extraction root into standard per-case directories.

    For each case in root/data/, copies the matching seg file from root/labels/
    into the case directory. Returns the list of normalised case directories.

    After this step every case directory contains all 5 files (4 modalities + seg)
    and the rest of the pipeline is format-agnostic.
    """
    data_dir   = root / "data"
    labels_dir = root / "labels"
    case_dirs: List[Path] = []

    for case_dir in sorted(data_dir.iterdir()):
        if not case_dir.is_dir():
            continue

        # Destination in our cache, mirrored from the data/ subdir
        dest_dir = case_extract_root / case_dir.name
        dest_dir.mkdir(parents=True, exist_ok=True)

        # Copy modality files if not already there
        for nii_file in case_dir.glob("*.nii.gz"):
            dest_file = dest_dir / nii_file.name
            if not dest_file.exists():
                shutil.copy2(nii_file, dest_file)

        # Find and copy the matching seg file from labels/
        # Seg filename pattern: <case_id>-seg.nii.gz
        seg_src = labels_dir / f"{case_dir.name}-seg.nii.gz"
        if seg_src.exists():
            dest_seg = dest_dir / seg_src.name
            if not dest_seg.exists():
                shutil.copy2(seg_src, dest_seg)
        else:
            print(f"  [warn] No seg file found for {case_dir.name}, skipping.")
            continue

        case_dirs.append(dest_dir)

    return case_dirs


def extract_case_from_tar(
    tar_path: str | Path,
    extract_root: str | Path = ".cache/brats",
    force: bool = False,
) -> Path:
    """Extract a BraTS 2021-style single-case .tar archive.

    Skips re-extraction if the target directory already has .nii.gz files.
    Returns the case directory.
    """
    tar_path     = Path(tar_path)
    extract_root = Path(extract_root)
    case_root    = extract_root / tar_path.stem
    case_root.mkdir(parents=True, exist_ok=True)

    # Skip if already extracted
    if list(case_root.glob("*.nii.gz")) and not force:
        return case_root

    with tarfile.open(tar_path, "r:*") as archive:
        archive.extractall(case_root)

    if not list(case_root.glob("**/*.nii.gz")):
        raise FileNotFoundError(f"No NIfTI files found after extracting {tar_path}")
    return case_root


def extract_brats2024_archive(
    tar_path: str | Path,
    extract_root: str | Path = ".cache/brats",
    force: bool = False,
) -> List[Path]:
    """Extract a BraTS 2024 GLI multi-case .tar.gz archive.

    Extracts to a staging area, detects the data/labels structure, normalises
    each case (copies seg into its case dir), and returns a list of case dirs.
    """
    tar_path     = Path(tar_path)
    extract_root = Path(extract_root)

    # Staging dir: strip all suffixes (.tar.gz → just the base stem)
    stem = tar_path.name
    for suffix in (".tar.gz", ".tgz", ".tar"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    staging_dir = extract_root / f"{stem}_staging"

    # Only re-extract if forced or staging dir is empty
    if not staging_dir.exists() or force:
        staging_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tar_path, "r:*") as archive:
            archive.extractall(staging_dir)

    # Detect BraTS 2024 root (may be nested one level deep)
    brats2024_root: Path | None = None
    if _is_brats2024_root(staging_dir):
        brats2024_root = staging_dir
    else:
        for subdir in staging_dir.iterdir():
            if subdir.is_dir() and _is_brats2024_root(subdir):
                brats2024_root = subdir
                break

    if brats2024_root is None:
        raise ValueError(
            f"{tar_path} does not appear to be a BraTS 2024 GLI archive "
            f"(expected data/ and labels/ subdirectories)."
        )

    case_extract_root = extract_root / stem
    case_extract_root.mkdir(parents=True, exist_ok=True)
    return _normalize_brats2024_cases(brats2024_root, case_extract_root)


# ---------------------------------------------------------------------------
# Case discovery
# ---------------------------------------------------------------------------

def case_id_from_path(case_path: str | Path) -> str:
    """Extract a human-readable case ID from a path."""
    case_path = Path(case_path)
    if case_path.is_file():
        return case_path.stem
    return case_path.name


def _find_by_alias(files: Iterable[Path], aliases: Iterable[str]) -> Path:
    """Find the first file whose lowercase name ends with any of the given aliases."""
    lower_to_file = {f.name.lower(): f for f in files}
    for name, path in lower_to_file.items():
        for suffix in aliases:
            if name.endswith(suffix):
                return path
    raise FileNotFoundError(f"Could not find file for aliases: {aliases}")


def _dir_has_required_modalities(case_dir: Path) -> bool:
    """Return True if case_dir contains all 5 required BraTS NIfTI files."""
    nii_files = list(case_dir.glob("*.nii.gz"))
    if not nii_files:
        return False
    names = [p.name.lower() for p in nii_files]
    for aliases in MODALITY_ALIASES.values():
        if not any(any(name.endswith(s) for s in aliases) for name in names):
            return False
    return True


def discover_case_paths(input_path: str | Path, max_cases: int = 0) -> List[Path]:
    """Find all BraTS cases under input_path. Supports both BraTS 2021 and 2024.

    Handles:
    - A single .tar file (BraTS 2021, one case)
    - A single .tar.gz file (BraTS 2024 GLI, multiple cases)
    - A directory containing .tar / .tar.gz archives
    - A directory of already-extracted case directories

    Returns a deduplicated list of per-case directories ready for loading.
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    cases: List[Path] = []

    if input_path.is_file():
        # Single archive provided directly
        if input_path.suffix in (".gz",) or input_path.name.endswith(".tar.gz"):
            # BraTS 2024 GLI multi-case archive
            cases.extend(extract_brats2024_archive(input_path))
        else:
            # BraTS 2021 single-case .tar
            cases.append(input_path)
    else:
        # Directory: check if it's already an extracted case
        if _dir_has_required_modalities(input_path):
            cases.append(input_path)

        # Find all .tar archives (BraTS 2021 style, one case each)
        tar_paths = sorted(
            p for p in input_path.rglob("*.tar")
            if not p.name.endswith(".tar.gz")
            and not any(part.startswith(".") for part in p.parts)
        )
        cases.extend(tar_paths)

        # Find all .tar.gz archives (BraTS 2024 style, multi-case)
        tar_gz_paths = sorted(
            p for p in input_path.rglob("*.tar.gz")
            if not any(part.startswith(".") for part in p.parts)
        )
        for tgz in tar_gz_paths:
            try:
                extracted = extract_brats2024_archive(tgz)
                cases.extend(extracted)
            except ValueError as e:
                print(f"  [warn] Skipping {tgz.name}: {e}")

        # Find already-extracted case directories
        case_dirs = sorted(
            p for p in input_path.rglob("*")
            if p.is_dir()
            and not any(part.startswith(".") for part in p.parts)
            and _dir_has_required_modalities(p)
        )
        cases.extend(case_dirs)

    # Deduplicate by case ID
    unique: List[Path] = []
    seen: set[str] = set()
    for case in cases:
        cid = case_id_from_path(case).lower()
        if cid not in seen:
            unique.append(case)
            seen.add(cid)

    if max_cases and max_cases > 0:
        unique = unique[:max_cases]

    if not unique:
        raise FileNotFoundError(f"No BraTS cases found in: {input_path}")
    return unique


def resolve_case_dir(case_path: str | Path) -> Path:
    """Resolve a case path to a directory containing .nii.gz files.

    For a BraTS 2021 .tar file: extracts it and returns the case directory.
    For an already-extracted directory: returns it directly.
    Note: BraTS 2024 .tar.gz files are handled in discover_case_paths;
    by the time resolve_case_dir is called, they are already extracted.
    """
    case_path = Path(case_path)
    if case_path.is_file() and case_path.suffix == ".tar":
        return extract_case_from_tar(case_path)
    if case_path.is_dir():
        return case_path
    raise FileNotFoundError(f"Case path does not exist or unsupported type: {case_path}")


# ---------------------------------------------------------------------------
# Volume loading and preprocessing
# ---------------------------------------------------------------------------

def load_case_volumes(case_path: str | Path) -> Dict[str, np.ndarray]:
    """Load all 5 NIfTI volumes for a BraTS case into memory.

    Works for both BraTS 2021 and 2024 cases (after normalisation).
    Returns a dict: {'t1', 't1ce', 't2', 'flair', 'seg'} → (H, W, D) float32 arrays.
    """
    case_dir  = resolve_case_dir(case_path)
    nii_files = list(case_dir.rglob("*.nii.gz"))
    if not nii_files:
        raise FileNotFoundError(f"No .nii.gz files found in {case_dir}")

    volumes: Dict[str, np.ndarray] = {}
    for key, aliases in MODALITY_ALIASES.items():
        file_path      = _find_by_alias(nii_files, aliases)
        volumes[key]   = nib.load(str(file_path)).get_fdata().astype(np.float32)

    # Sanity check: all modalities must have the same spatial shape
    shape = volumes["seg"].shape
    for key, vol in volumes.items():
        if vol.shape != shape:
            raise ValueError(f"Shape mismatch: {key} has {vol.shape}, expected {shape}")

    return volumes


def normalize_zscore(volume: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Z-score normalise a volume: subtract mean, divide by std.

    Applied per modality per volume (not globally) because MRI intensities
    are not calibrated across scanners or acquisition protocols.
    """
    mean = float(np.mean(volume))
    std  = float(np.std(volume))
    return ((volume - mean) / (std + eps)).astype(np.float32)


def normalize_modalities(
    volumes: Dict[str, np.ndarray],
    keys: Tuple[str, ...] = ("t1", "t1ce", "t2", "flair"),
) -> Dict[str, np.ndarray]:
    """Apply z-score normalisation to each MRI modality independently.

    The segmentation mask is left untouched (it contains integer labels).
    """
    normalised = dict(volumes)
    for key in keys:
        normalised[key] = normalize_zscore(normalised[key])
    return normalised


def to_binary_mask(seg: np.ndarray) -> np.ndarray:
    """Convert multi-class BraTS labels to a single binary tumor mask.

    BraTS label convention (both 2021 and 2024):
      0 = background
      1 = necrotic core (NCR)
      2 = peritumoral edema (ED)
      3 or 4 = enhancing tumor (ET)   ← BraTS 2024 uses 3 instead of 4

    All non-zero labels are merged into tumor class (1).
    """
    return (seg > 0).astype(np.float32)


def find_tumor_slices(mask_3d: np.ndarray, min_tumor_pixels: int = 1) -> List[int]:
    """Return axial slice indices that contain at least min_tumor_pixels of tumor.

    Filters out blank slices (no tumor) to avoid extreme class imbalance.
    mask_3d shape: (H, W, D) where D is the number of axial slices.
    """
    return [
        idx for idx in range(mask_3d.shape[-1])
        if int(np.sum(mask_3d[:, :, idx])) >= min_tumor_pixels
    ]


def build_slices(
    volumes: Dict[str, np.ndarray],
    only_tumor: bool = True,
    min_tumor_pixels: int = 1,
    return_indices: bool = False,
) -> Tuple[np.ndarray, np.ndarray] | Tuple[np.ndarray, np.ndarray, List[int]]:
    """Convert 3D volumes into a stack of 2D axial slices.

    For each selected axial index:
      - Stacks 4 modality slices → (4, H, W) input tensor
      - Takes binary mask slice  → (1, H, W) target tensor

    Returns:
        X: (N, 4, H, W) float32
        y: (N, 1, H, W) float32
        (optional) slice_indices: list of selected axial indices
    """
    seg   = to_binary_mask(volumes["seg"])
    depth = seg.shape[-1]

    slice_indices = (
        find_tumor_slices(seg, min_tumor_pixels=min_tumor_pixels)
        if only_tumor else list(range(depth))
    )
    if not slice_indices:
        raise ValueError("No slices selected. Check mask or min_tumor_pixels.")

    x_slices, y_slices = [], []
    for idx in slice_indices:
        image = np.stack(
            [volumes["t1"][:, :, idx],
             volumes["t1ce"][:, :, idx],
             volumes["t2"][:, :, idx],
             volumes["flair"][:, :, idx]],
            axis=0,
        ).astype(np.float32)                          # (4, H, W)
        mask = seg[:, :, idx][None, :, :].astype(np.float32)  # (1, H, W)
        x_slices.append(image)
        y_slices.append(mask)

    X = np.stack(x_slices, axis=0).astype(np.float32)   # (N, 4, H, W)
    y = np.stack(y_slices, axis=0).astype(np.float32)   # (N, 1, H, W)

    if return_indices:
        return X, y, slice_indices
    return X, y


def build_dataset_from_cases(
    case_paths: Sequence[str | Path],
    only_tumor: bool = True,
    min_tumor_pixels: int = 1,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, int | str]]]:
    """Load and concatenate 2D slice data from multiple BraTS cases.

    Each case is normalised independently (per-case z-score), then all
    slices are concatenated into a single dataset array.

    Returns:
        X_cat:   (N_total, 4, H, W) float32
        y_cat:   (N_total, 1, H, W) float32
        summary: per-case dict with case_id, num_slices, num_tumor_slices
    """
    x_all:   List[np.ndarray] = []
    y_all:   List[np.ndarray] = []
    summary: List[Dict[str, int | str]] = []

    for case in case_paths:
        case_id = case_id_from_path(case)
        volumes = normalize_modalities(load_case_volumes(case))
        X, y, indices = build_slices(
            volumes,
            only_tumor=only_tumor,
            min_tumor_pixels=min_tumor_pixels,
            return_indices=True,
        )
        x_all.append(X)
        y_all.append(y)
        summary.append({
            "case_id":          case_id,
            "num_slices":       int(X.shape[0]),
            "num_tumor_slices": int(len(indices)),
        })

    if not x_all:
        raise ValueError("No slices were built from the provided case paths.")

    X_cat = np.concatenate(x_all, axis=0).astype(np.float32)
    y_cat = np.concatenate(y_all, axis=0).astype(np.float32)
    return X_cat, y_cat, summary
