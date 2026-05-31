"""Small image-sequence reader used by preprocessing simulation scripts."""

from __future__ import annotations

import os
import random
import re
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, UnidentifiedImageError


def _natural_key(value: str | os.PathLike[str]) -> list[int | str]:
    text = str(value)
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", text)
        if part
    ]


class VideoReader:
    """Read one subdirectory of image frames as an ``(N, H, W, C)`` array."""

    def __init__(
        self,
        root: str,
        crop_size: Optional[tuple[int, int]] = None,
        random_crop: bool = True,
        max_frames: Optional[int] = None,
        exts: Optional[list[str]] = None,
    ) -> None:
        self.root = root
        self.crop_size = crop_size
        self.random_crop = random_crop
        self.max_frames = max_frames
        self.exts = tuple(ext.lower() for ext in (exts or [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]))

        root_path = Path(root)
        if not root_path.is_dir():
            raise FileNotFoundError(f"VideoReader root does not exist: {root}")

        self.scenes = [
            path.name
            for path in sorted(root_path.iterdir(), key=_natural_key)
            if path.is_dir()
        ]

    def __len__(self) -> int:
        return len(self.scenes)

    def _list_scene_files(self, scene: str) -> list[str]:
        scene_dir = Path(self.root) / scene
        files = [
            path
            for path in sorted(scene_dir.iterdir(), key=_natural_key)
            if path.is_file() and path.suffix.lower() in self.exts and not path.name.startswith(".")
        ]
        if self.max_frames is not None:
            files = files[: self.max_frames]
        return [str(path) for path in files]

    def _load_image(self, path: str) -> np.ndarray | None:
        try:
            with Image.open(path) as image:
                return np.array(image.convert("RGB"))
        except UnidentifiedImageError:
            return None
        except OSError:
            return None

    def _random_crop_coords(self, h: int, w: int, crop_h: int, crop_w: int) -> tuple[int, int]:
        if h == crop_h and w == crop_w:
            return 0, 0
        top = random.randint(0, h - crop_h) if h > crop_h else 0
        left = random.randint(0, w - crop_w) if w > crop_w else 0
        return top, left

    def _apply_crop(self, imgs: list[np.ndarray]) -> list[np.ndarray]:
        if self.crop_size is None:
            return imgs

        crop_h, crop_w = self.crop_size
        h, w = imgs[0].shape[:2]
        if crop_h > h or crop_w > w:
            raise ValueError(f"crop_size {(crop_h, crop_w)} exceeds source size {(h, w)}")

        top = (h - crop_h) // 2
        left = (w - crop_w) // 2
        if self.random_crop:
            top, left = self._random_crop_coords(h, w, crop_h, crop_w)

        return [img[top : top + crop_h, left : left + crop_w, :] for img in imgs]

    def __getitem__(self, index: int) -> np.ndarray:
        if index < 0 or index >= len(self.scenes):
            raise IndexError("scene index out of range")

        scene = self.scenes[index]
        files = self._list_scene_files(scene)
        if not files:
            raise RuntimeError(f"no image files found in scene: {scene}")

        imgs = [self._load_image(path) for path in files]
        valid_imgs = [img for img in imgs if img is not None]
        if not valid_imgs:
            raise RuntimeError(f"no valid image files found in scene: {scene}")

        return np.stack(self._apply_crop(valid_imgs), axis=0)
