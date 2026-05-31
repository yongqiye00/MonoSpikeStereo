"""Datasets for NPZ-based MonoSpikeStereo training and evaluation."""

import glob
import os
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .augmentation import StereoRobustAug


def _has_array(data: object, key: str) -> bool:
    if hasattr(data, "files"):
        return key in data.files
    if isinstance(data, dict):
        return key in data and data[key] is not None
    return False


def _load_mt_clip(data: object, start_idx: int, length: int) -> np.ndarray:
    """Load the stored model reference as M_t / Q."""
    if _has_array(data, "mt"):
        return np.asarray(data["mt"][start_idx : start_idx + length])

    raise KeyError("NPZ must contain 'mt'.")


class RealStereoDataset(Dataset):
    """Dataset that returns temporal clips from a single NPZ file.

    Each NPZ is expected to contain arrays with shape (T, H, W).

    Returned tensors:
        left:      (T, 1, H, W) -- observed sequence normalized to [0, 1]
        right:     (T, 1, H, W) -- observed sequence normalized to [0, 1]
        f_clip:    (T,)         -- per-frame LCD modulation values (float32)
        target_a:  (T, 1, H, W) -- reference frames for the left camera in [0, 1]
        target_b:  (T, 1, H, W) -- reference frames for the right camera in [0, 1]
    """

    def __init__(
        self,
        npz_path: str,
        sequence_length: int = 5,
        split: str = "train",
        train_frac: float = 0.8,
        val_frac: float = 0.1,
        transforms: Optional[object] = None,
        crop_size: Optional[Tuple[int, int]] = None,
        real: bool = False,
    ) -> None:
        """
        Args:
            npz_path: Path to the .npz file.
            sequence_length: Number of temporal frames per sample (odd, e.g. 5).
            split: One of 'train', 'val' or 'test' (test uses remaining indices).
            train_frac: Fraction of valid indices used for training.
            val_frac: Fraction used for validation.
            transforms: Optional callable applied to the returned tensors.
        """
        super().__init__()
        assert sequence_length >= 1 and sequence_length % 2 == 1, (
            "sequence_length must be odd"
        )
        self.sequence_length = sequence_length
        self.half = sequence_length // 2
        self.split = split
        self.transforms = transforms
        # crop_size: (H, W) or None. If provided, training uses random crop, val/test uses center crop
        if crop_size is not None:
            assert isinstance(crop_size, (list, tuple)) and len(crop_size) == 2
        self.crop_size = tuple(crop_size) if crop_size is not None else None
        self.real = real

        # Support passing either a single .npz file or a directory containing many .npz files
        paths = []
        if os.path.isdir(npz_path):
            # collect all .npz files (sorted for deterministic behavior)
            # recursive search for .npz files in directory and subdirectories
            paths = sorted(
                glob.glob(os.path.join(npz_path, "**", "*.npz"), recursive=True)
            )
            # Exclude files whose path contains "city" (case-insensitive).
            # paths = [p for p in paths if "city" not in p.lower()]

            if len(paths) == 0:
                raise ValueError(f"No .npz files found under: {npz_path}")
        elif os.path.isfile(npz_path) and npz_path.endswith(".npz"):
            paths = [npz_path]
        else:
            if not os.path.exists(npz_path):
                raise ValueError(f"npz_path does not exist: {npz_path}")
            raise ValueError(f"npz_path must be a .npz file or directory: {npz_path}")
        paths = paths[:]
        self.paths = paths
        # Record paths and per-file metadata, but do NOT load full arrays into memory here.

    def __len__(self) -> int:
        return len(self.paths)

    def _to_tensor(
        self, arr: np.ndarray, add_channel: bool = True, normalize: bool = True
    ) -> torch.Tensor:
        """Convert numpy array to torch tensor with optional [0,1] normalization.

        uint8 inputs are divided by 255 when `normalize` is True. Floating-point
        arrays are left unchanged unless values exceed 1, in which case they are
        scaled down by 255 to avoid oversized magnitudes.
        """
        arr = arr.astype(np.float32)
        # if normalize:
        #     if arr.max() > 1.0:
        #         arr = arr / 255.0
        tensor = torch.from_numpy(arr)
        if add_channel:
            tensor = tensor.unsqueeze(1)  # (T, 1, H, W) or (1, H, W) -> (1, 1, H, W)
        return tensor

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        data = np.load(self.paths[0], mmap_mode="r")
        left_np = data["left"].clip(0, 1)
        right_np = data["right"].clip(0, 1)
        L = min(self.sequence_length, left_np.shape[0])
        if left_np.shape[0] < L:
            raise ValueError(
                f"File {self.paths[0]} has only {left_np.shape[0]} frames, less than sequence_length {L}"
            )

        start_idx = np.random.randint(0, left_np.shape[0] - L + 1)
        left_np = left_np[start_idx : start_idx + L][:, :240]
        right_np = right_np[start_idx : start_idx + L][:, :240]
        mt = _load_mt_clip(data, start_idx, L)[:, :240]
        data.close()

        # to tensors
        left = self._to_tensor(left_np, add_channel=True)  # (T,1,H,W)
        right = self._to_tensor(right_np, add_channel=True)  # (T,1,H,W)
        mt = self._to_tensor(mt, add_channel=True)  # (T,1,H,W)
        const = 2
        sample = {
            "left": (left * const),  # (T,1,H,W)
            "right": (right * const) * 2,  # (T,1,H,W)
            "mt": mt,  # (T,1,H,W)
            "target_a": mt * const,  # (1,H,W)
            "target_b": right,  # (1,H,W)
            "f_clip": torch.zeros(L, dtype=torch.float32),  # (T,)
            "intensity": torch.zeros_like(left),  # (T,1,H,W)
            "threshold": torch.tensor(const, dtype=torch.float32),  # scalar
        }
        return sample


def augment(
    inp, factor_param=(1, 0.1), move_range=(0.5, 1.5), prob_salt=0.01
) -> np.ndarray:
    """
    Add Gaussian noise and salt-and-pepper noise to a numpy array.
    inp: numpy.ndarray with shape (T,H,W) or (T,1,H,W)
    Returns the perturbed numpy array.
    """
    arr = inp.copy()

    factor = np.random.normal(
        factor_param[0], factor_param[1], size=(arr.shape[0], 1, 1)
    )  # (T,1,1)
    factor = np.clip(factor, move_range[0], move_range[1]).astype(arr.dtype)
    arr = arr * factor

    # Gaussian noise.
    sigma = 0.01  # * (arr.max() - arr.min() + 1e-6)
    arr += np.random.normal(0, sigma, size=arr.shape).astype(arr.dtype)

    # Salt-and-pepper noise.
    prob = prob_salt  # Salt-and-pepper probability.
    salt_mask = np.random.rand(*arr.shape) < (prob / 2)
    pepper_mask = np.random.rand(*arr.shape) < (prob / 2)
    arr[salt_mask] = arr.max()
    arr[pepper_mask] = arr.min()

    # Brightness perturbation.
    # if np.random.rand() < 0.5:

    return arr


class StereoSimulationDataset(Dataset):
    def augment(
        self, inp, factor_param=(1, 0.1), move_range=(0.5, 1.5), prob_salt=0.01
    ) -> np.ndarray:
        """
        Add Gaussian noise and salt-and-pepper noise to a numpy array.
        inp: numpy.ndarray with shape (T,H,W) or (T,1,H,W)
        Returns the perturbed numpy array.
        """
        arr = inp.copy()

        factor = np.random.normal(
            factor_param[0], factor_param[1], size=(arr.shape[0], 1, 1)
        )  # (T,1,1)
        factor = np.clip(factor, move_range[0], move_range[1]).astype(arr.dtype)
        arr = arr * factor

        # Gaussian noise.
        sigma = 0.01  # * (arr.max() - arr.min() + 1e-6)
        arr += np.random.normal(0, sigma, size=arr.shape).astype(arr.dtype)

        # Salt-and-pepper noise.
        prob = prob_salt  # Salt-and-pepper probability.
        salt_mask = np.random.rand(*arr.shape) < (prob / 2)
        pepper_mask = np.random.rand(*arr.shape) < (prob / 2)
        arr[salt_mask] = arr.max()
        arr[pepper_mask] = arr.min()

        # Brightness perturbation.
        # if np.random.rand() < 0.5:

        return arr

    def __init__(
        self,
        npz_path: str,
        sequence_length: int = 5,
        split: str = "train",
        crop_size: Optional[Tuple[int, int]] = None,
        clip_stride: int = 1,
    ) -> None:
        super().__init__()
        # assert sequence_length >= 1 and sequence_length % 2 == 1, "sequence_length must be odd"
        self.sequence_length = sequence_length
        self.half = sequence_length // 2
        self.split = split
        self.crop_size = crop_size
        self.paths = []
        self.clip_stride = max(1, int(clip_stride))
        self.samples: list[tuple[int, int]] = []
        self.file_lengths: list[int] = []

        self.stereo_augmenter = StereoRobustAug(
            max_shift=2, max_angle=0.2, prob=0.5, apply_shift=True, apply_rotate=True
        )
        if os.path.isdir(npz_path):
            self.paths = sorted(
                glob.glob(os.path.join(npz_path, "**", "*.npz"), recursive=True)
            )
            if self.split == "metrics":
                self.paths = [
                    p
                    for p in self.paths
                    if "group_" in p
                    and int(p.split("group_")[-1].split("/")[0][:-4]) % 15 == 0
                ]

        for path_idx, path in enumerate(self.paths):
            with np.load(path, mmap_mode="r") as data:
                length = int(data["left"].shape[0])
            self.file_lengths.append(length)
            clip_len = self._clip_length(length)
            max_start = max(0, length - clip_len)
            if self.split == "train":
                starts = range(0, max_start + 1, self.clip_stride)
                self.samples.extend((path_idx, start) for start in starts)
            else:
                self.samples.append((path_idx, 0))

        print(
            f"{split} dataset: {len(self.paths)} npz files, "
            f"{len(self.samples)} clips found in {npz_path}"
        )

    def _clip_length(self, available_frames: int) -> int:
        return max(1, min(self.sequence_length, available_frames))

    def _to_tensor(
        self, arr: np.ndarray, add_channel: bool = True, normalize: bool = True
    ) -> torch.Tensor:
        """Convert numpy array to torch tensor with optional [0,1] normalization.

        uint8 inputs are divided by 255 when `normalize` is True. Floating-point
        arrays are left unchanged unless values exceed 1, in which case they are
        scaled down by 255 to avoid oversized magnitudes.
        """
        arr = arr.astype(np.float32)
        # if normalize:
        #     if arr.max() > 1.0:
        #         arr = arr / 255.0
        tensor = torch.from_numpy(arr)
        if add_channel:
            tensor = tensor.unsqueeze(1)  # (T, 1, H, W) or (1, H, W) -> (1, 1, H, W)
        return tensor

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        path_idx, base_start_idx = self.samples[idx]
        path = self.paths[path_idx]
        data = np.load(path, mmap_mode="r")
        left_np = data["left"]
        right_np = data["right"]
        left_gt = data["vid_left"]
        right_gt = data["vid_right"]
        threshold = data["threshold"].item()

        L = self._clip_length(left_np.shape[0])
        max_start = max(0, left_np.shape[0] - L)
        start_idx = base_start_idx
        if self.split == "train" and self.clip_stride > 1:
            jitter_limit = min(self.clip_stride, max_start - base_start_idx + 1)
            if jitter_limit > 1:
                start_idx = base_start_idx + np.random.randint(0, jitter_limit)
        left_np = left_np[start_idx : start_idx + L]
        right_np = right_np[start_idx : start_idx + L]
        mt = _load_mt_clip(data, start_idx, L)
        left_gt = left_gt[start_idx : start_idx + L]
        right_gt = right_gt[start_idx : start_idx + L]
        data.close()

        # to tensors
        if self.crop_size is not None:
            H, W = left_np.shape[1:3]
            ch, cw = self.crop_size
            if self.split == "train":
                top = np.random.randint(0, H - ch + 1)
                left = np.random.randint(0, W - cw + 1)
            else:
                top = (H - ch) // 2
                left = (W - cw) // 2
            left_np = left_np[:, top : top + ch, left : left + cw]
            right_np = right_np[:, top : top + ch, left : left + cw]
            mt = mt[:, top : top + ch, left : left + cw]
            left_gt = left_gt[:, top : top + ch, left : left + cw]
            right_gt = right_gt[:, top : top + ch, left : left + cw]

        if self.split == "train":
            # data augmentation can be added here

            if np.random.rand() < 0.5:
                left_np = np.flip(left_np, axis=1)
                right_np = np.flip(right_np, axis=1)
                left_gt = np.flip(left_gt, axis=1)
                right_gt = np.flip(right_gt, axis=1)
                mt = np.flip(mt, axis=1)

            if np.random.rand() < 0.5:
                left_np = augment(left_np, factor_param=(1, 0.1), move_range=(0.9, 1.1))
                right_np = augment(
                    right_np,
                    factor_param=(0.4, 0.3),
                    move_range=(0.1, 0.8),
                    prob_salt=0.05,
                )

        left = self._to_tensor(left_np, add_channel=True)  # (T,1,H,W)
        right = self._to_tensor(right_np, add_channel=True)  # (T,1,H,W)

        mt = self._to_tensor(mt, add_channel=True)  # (T,1,H,W)
        left_gt = self._to_tensor(left_gt, add_channel=True)  # (T,1,H,W)
        right_gt = self._to_tensor(right_gt, add_channel=True)  # (T,1,H,W)
        threshold_value = float(threshold)
        threshold = torch.tensor(threshold_value, dtype=torch.float32)
        # left, right, right_gt = self.stereo_augmenter(left, right, right_gt)

        sample = {
            "left": left.clamp(min=0.0, max=threshold_value),  # (T,1,H,W)
            "right": right.clamp(min=0.0, max=threshold_value),  # (T,1,H,W)
            "mt": mt,  # (T,1,H,W)
            "target_a": left_gt,  # (1,H,W)
            "target_b": right_gt,  # (1,H,W)
            "f_clip": torch.zeros(L, dtype=torch.float32),  # (T,)
            "intensity": torch.zeros_like(left),  # (T,1,H,W)
            "threshold": threshold,  # scalar
        }

        return sample

if __name__ == "__main__":
    raise SystemExit(
        "Dataset smoke tests need a local npz dataset. "
        "Use train.py --config configs/train/train.yaml --npz_path <path>."
    )
