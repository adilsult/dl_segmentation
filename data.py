from __future__ import annotations

import tarfile
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import nibabel as nib
import numpy as np


MODALITY_ALIASES = {
    "t1": ("_t1.nii.gz", "_t1n.nii.gz", "-t1.nii.gz", "-t1n.nii.gz"),
    "t1ce": ("_t1ce.nii.gz", "_t1c.nii.gz", "-t1ce.nii.gz", "-t1c.nii.gz"),
    "t2": ("_t2.nii.gz", "_t2w.nii.gz", "-t2.nii.gz", "-t2w.nii.gz"),
    "flair": ("_flair.nii.gz", "_t2f.nii.gz", "-flair.nii.gz", "-t2f.nii.gz"),
    "seg": ("_seg.nii.gz", "-seg.nii.gz"),
}


def extract_case_from_tar(
    tar_path: str | Path,
    extract_root: str | Path = ".cache/brats",
    force: bool = False,
) -> Path:
    tar_path = Path(tar_path)
    extract_root = Path(extract_root)
    case_root = extract_root / tar_path.stem
    case_root.mkdir(parents=True, exist_ok=True)

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
    case_path = Path(case_path)
    if case_path.is_file():
        return case_path.stem
    return case_path.name


def _find_by_alias(files: Iterable[Path], aliases: Iterable[str]) -> Path:
    lower_to_file = {f.name.lower(): f for f in files}
    for name, path in lower_to_file.items():
        for suffix in aliases:
            if name.endswith(suffix):
                return path
    raise FileNotFoundError(f"Could not find file for aliases: {aliases}")


def _dir_has_required_modalities(case_dir: Path) -> bool:
    nii_files = [p for p in case_dir.glob("*.nii.gz")]
    if not nii_files:
        return False

    names = [p.name.lower() for p in nii_files]
    for aliases in MODALITY_ALIASES.values():
        if not any(any(name.endswith(suffix) for suffix in aliases) for name in names):
            return False
    return True


def discover_case_paths(input_path: str | Path, max_cases: int = 0) -> List[Path]:
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    cases: List[Path] = []
    if input_path.is_file():
        cases = [input_path]
    else:
        # If the folder itself is a case directory, include it first.
        if _dir_has_required_modalities(input_path):
            cases.append(input_path)

        tar_paths = sorted(
            p for p in input_path.rglob("*.tar") if not any(part.startswith(".") for part in p.parts)
        )
        cases.extend(tar_paths)

        case_dirs = sorted(
            p
            for p in input_path.rglob("*")
            if p.is_dir() and not any(part.startswith(".") for part in p.parts) and _dir_has_required_modalities(p)
        )
        cases.extend(case_dirs)

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
    case_path = Path(case_path)
    if case_path.is_file() and case_path.suffix == ".tar":
        return extract_case_from_tar(case_path)
    if case_path.is_dir():
        return case_path
    raise FileNotFoundError(f"Case path does not exist or unsupported type: {case_path}")


def load_case_volumes(case_path: str | Path) -> Dict[str, np.ndarray]:
    case_dir = resolve_case_dir(case_path)
    nii_files = [p for p in case_dir.rglob("*.nii.gz")]
    if not nii_files:
        raise FileNotFoundError(f"No .nii.gz files found in {case_dir}")

    volumes: Dict[str, np.ndarray] = {}
    for key, aliases in MODALITY_ALIASES.items():
        file_path = _find_by_alias(nii_files, aliases)
        volumes[key] = nib.load(str(file_path)).get_fdata().astype(np.float32)

    shape = volumes["seg"].shape
    for key, volume in volumes.items():
        if volume.shape != shape:
            raise ValueError(f"Shape mismatch: {key} has {volume.shape}, expected {shape}")

    return volumes


def normalize_zscore(volume: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    mean = float(np.mean(volume))
    std = float(np.std(volume))
    return ((volume - mean) / (std + eps)).astype(np.float32)


def normalize_modalities(
    volumes: Dict[str, np.ndarray],
    keys: Tuple[str, ...] = ("t1", "t1ce", "t2", "flair"),
) -> Dict[str, np.ndarray]:
    normalized = dict(volumes)
    for key in keys:
        normalized[key] = normalize_zscore(normalized[key])
    return normalized


def to_binary_mask(seg: np.ndarray) -> np.ndarray:
    return (seg > 0).astype(np.float32)


def find_tumor_slices(mask_3d: np.ndarray, min_tumor_pixels: int = 1) -> List[int]:
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
    seg = to_binary_mask(volumes["seg"])
    depth = seg.shape[-1]

    if only_tumor:
        slice_indices = find_tumor_slices(seg, min_tumor_pixels=min_tumor_pixels)
    else:
        slice_indices = list(range(depth))

    if not slice_indices:
        raise ValueError("No slices selected. Check mask or min_tumor_pixels.")

    x_slices = []
    y_slices = []
    for idx in slice_indices:
        image = np.stack(
            [
                volumes["t1"][:, :, idx],
                volumes["t1ce"][:, :, idx],
                volumes["t2"][:, :, idx],
                volumes["flair"][:, :, idx],
            ],
            axis=0,
        ).astype(np.float32)
        mask = seg[:, :, idx][None, :, :].astype(np.float32)
        x_slices.append(image)
        y_slices.append(mask)

    X = np.stack(x_slices, axis=0).astype(np.float32)
    y = np.stack(y_slices, axis=0).astype(np.float32)

    if return_indices:
        return X, y, slice_indices
    return X, y


def build_dataset_from_cases(
    case_paths: Sequence[str | Path],
    only_tumor: bool = True,
    min_tumor_pixels: int = 1,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, int | str]]]:
    x_all: List[np.ndarray] = []
    y_all: List[np.ndarray] = []
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
        summary.append(
            {
                "case_id": case_id,
                "num_slices": int(X.shape[0]),
                "num_tumor_slices": int(len(indices)),
            }
        )

    if not x_all:
        raise ValueError("No slices were built from the provided case paths.")

    X_cat = np.concatenate(x_all, axis=0).astype(np.float32)
    y_cat = np.concatenate(y_all, axis=0).astype(np.float32)
    return X_cat, y_cat, summary
