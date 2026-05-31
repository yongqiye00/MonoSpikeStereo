"""Shared image and IO helpers for preprocessing."""

from __future__ import annotations

import os
import re
from glob import glob
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}


def ensure_dir(path: str | os.PathLike[str]) -> None:
    """Create a directory if it does not already exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


def center_crop(
    arr: np.ndarray,
    crop_size: tuple[int, int] | None,
) -> np.ndarray:
    """Center-crop the last two dimensions."""
    if crop_size is None:
        return arr

    target_h, target_w = crop_size
    if arr.ndim < 2:
        raise ValueError("array needs at least 2 dims to apply spatial crop")

    src_h, src_w = arr.shape[-2], arr.shape[-1]
    if target_h > src_h or target_w > src_w:
        raise ValueError(
            f"crop_size {crop_size} exceeds source dims {(src_h, src_w)}"
        )

    top = (src_h - target_h) // 2
    left = (src_w - target_w) // 2
    spatial_slices = (slice(top, top + target_h), slice(left, left + target_w))
    leading_slices = (slice(None),) * (arr.ndim - 2)
    return arr[leading_slices + spatial_slices]


def parse_crop(crop: str | None) -> tuple[int, int] | None:
    """Parse ``WxH`` into ``(H, W)``."""
    if crop is None:
        return None

    text = crop.strip().lower().replace(",", "x")
    if not text:
        return None

    parts = [part.strip() for part in text.split("x") if part.strip()]
    if len(parts) != 2:
        raise ValueError("crop format must be WxH, e.g. 128x128")

    width, height = int(parts[0]), int(parts[1])
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid crop size: {crop}")

    return height, width


def _stack_frames_or_none(frames: list[np.ndarray], root: str) -> np.ndarray | None:
    """Stack collected frames into a single array, returning None on mismatch."""
    if not frames:
        return None

    try:
        return np.stack(frames, axis=0)
    except ValueError as exc:
        print(f"Warning: depth frames have inconsistent shapes under {root}: {exc}")
        return None


def _natural_sort_key(path: str | os.PathLike[str]) -> list[int | str]:
    """Build a sort key that treats digit runs as integers."""
    text = str(path)
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", text)
        if part
    ]


def _iter_files(
    root: str | os.PathLike[str],
    pattern: str,
) -> list[str]:
    """Return naturally sorted recursive matches under a directory."""
    return sorted(
        glob(os.path.join(str(root), "**", pattern), recursive=True),
        key=_natural_sort_key,
    )


def load_depth_sequence(
    root: str | os.PathLike[str],
    max_frames: int | None = None,
) -> np.ndarray | None:
    """Recursively load `.npy` depth frames and stack them as `(N, H, W, ...)`."""
    if not os.path.isdir(root):
        return None

    npy_files = _iter_files(root, "*.npy")
    if not npy_files:
        return None

    frames: list[np.ndarray] = []
    for path in npy_files:
        try:
            frames.append(np.load(path))
        except Exception as exc:  # pragma: no cover - defensive data loading
            print(f"Warning: failed to load depth frame {path}: {exc}")
            continue

        if max_frames is not None and len(frames) >= max_frames:
            break

    return _stack_frames_or_none(frames, str(root))


def load_depth_sequence_png(
    root: str | os.PathLike[str],
    max_frames: int | None = None,
) -> np.ndarray | None:
    """Recursively load image-based depth frames."""
    if not os.path.isdir(root):
        return None

    image_files = _iter_files(root, "*.*")
    image_files = [
        path
        for path in image_files
        if Path(path).suffix.lower() in IMAGE_EXTENSIONS
    ]
    if not image_files:
        return None

    frames: list[np.ndarray] = []
    for path in image_files:
        try:
            with Image.open(path) as im:
                arr = np.array(im.convert("I"))
            frames.append(arr)
        except UnidentifiedImageError:  # pragma: no cover - corrupt input guard
            print(f"Warning: failed to load depth frame {path}: UnidentifiedImageError")
            continue
        except Exception as exc:  # pragma: no cover - defensive data loading
            print(f"Warning: failed to load depth frame {path}: {exc}")
            continue

        if max_frames is not None and len(frames) >= max_frames:
            break

    return _stack_frames_or_none(frames, str(root))


def nhwc_to_nhw_gray(arr: np.ndarray) -> np.ndarray:
    """Convert an NHWC RGB sequence into NHW grayscale."""
    if arr.ndim != 4:
        raise ValueError("expected NHWC ndarray")

    _, _, _, channels = arr.shape
    if channels == 1:
        return arr[:, :, :, 0].astype(np.float32, copy=False)

    weights = np.array([0.2989, 0.5870, 0.1140], dtype=np.float32)
    gray = np.tensordot(arr.astype(np.float32), weights, axes=([3], [0]))
    return gray.astype(np.float32, copy=False)


def mod_crop_01(img: np.ndarray, scale: int) -> np.ndarray:
    """Crop H/W so they are divisible by scale."""
    if scale <= 0:
        raise ValueError("scale must be positive")

    h, w = img.shape[-2:]
    h_cropped = h - (h % scale)
    w_cropped = w - (w % scale)

    if img.ndim == 2:
        return img[:h_cropped, :w_cropped]
    if img.ndim == 3:
        return img[:, :h_cropped, :w_cropped]

    raise ValueError("only 2D or 3D grayscale inputs are supported")


def downsample_bicubic_01(img: np.ndarray, scale: int) -> np.ndarray:
    """Downsample a 0-1 float sequence `(N, H, W)` using bicubic interpolation."""
    if img.ndim != 3:
        raise ValueError("expected input shape (N, H, W)")
    if scale <= 0:
        raise ValueError("scale must be positive")

    n, h, w = img.shape
    new_h, new_w = h // scale, w // scale
    if new_h <= 0 or new_w <= 0:
        raise ValueError(f"scale {scale} is too large for input shape {(n, h, w)}")

    return np.stack(
        [
            cv2.resize(img[i], (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            for i in range(n)
        ],
        axis=0,
    )


def hr_to_lr_01(hr_img: np.ndarray, scale: int) -> np.ndarray:
    """Convert a 0-1 HR grayscale sequence to LR using mod-crop + bicubic."""
    if scale not in {2, 3, 4}:
        raise ValueError("scale must be one of {2, 3, 4}")

    hr_cropped = mod_crop_01(hr_img, scale)
    return downsample_bicubic_01(hr_cropped, scale)


__all__ = [
    "IMAGE_EXTENSIONS",
    "center_crop",
    "downsample_bicubic_01",
    "ensure_dir",
    "hr_to_lr_01",
    "load_depth_sequence",
    "load_depth_sequence_png",
    "mod_crop_01",
    "nhwc_to_nhw_gray",
    "parse_crop",
]
