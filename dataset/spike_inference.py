"""Dataset and spike-splitting helpers for inference from raw spike sequences."""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from dataset.spike_reference import compute_mt_reference


def collect_npz_paths(npz_path: str, max_files: int = 0) -> list[str]:
    if os.path.isdir(npz_path):
        paths = sorted(glob.glob(os.path.join(npz_path, "**", "*.npz"), recursive=True))
    elif os.path.isfile(npz_path) and npz_path.endswith(".npz"):
        paths = [npz_path]
    else:
        if not os.path.exists(npz_path):
            raise ValueError(f"npz_path does not exist: {npz_path}")
        raise ValueError(f"npz_path must be a .npz file or directory: {npz_path}")

    if not paths:
        raise ValueError(f"No .npz files found under: {npz_path}")
    if max_files > 0:
        paths = paths[:max_files]
    return paths


def center_crop_array(arr: np.ndarray, crop_size: tuple[int, int] | None) -> np.ndarray:
    if crop_size is None:
        return arr
    ch, cw = crop_size
    h, w = arr.shape[-2:]
    if ch > h or cw > w:
        raise ValueError(f"crop_size {(ch, cw)} exceeds input spatial size {(h, w)}")
    top = (h - ch) // 2
    left = (w - cw) // 2
    return arr[(slice(None),) * (arr.ndim - 2) + (slice(top, top + ch), slice(left, left + cw))]


def load_stored_mt(data: np.lib.npyio.NpzFile, crop_size: tuple[int, int] | None) -> np.ndarray | None:
    if "mt" in data.files:
        return center_crop_array(np.asarray(data["mt"]), crop_size)
    return None


def read_reference_window_size(data: np.lib.npyio.NpzFile, window_size: int) -> int:
    for key in ("reference_window_size", "waveform_period", "lcd_period"):
        if key in data.files:
            return int(np.asarray(data[key]).item())
    return 4 * int(window_size)


def split_spikes_to_left_right(
    spk_seq: np.ndarray,
    lcd: np.ndarray,
    *,
    window_size: int,
    threshold: float,
    reg: float = 1e-3,
    show_progress: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct left/right observations from a spike sequence.

    This mirrors the preprocessing splitter used to create training NPZ files.
    The returned arrays are shaped ``(T, H, W)`` and already scaled by
    ``threshold``, matching the stored ``left`` / ``right`` payload convention.
    """
    spk_seq = np.asarray(spk_seq)
    lcd = np.asarray(lcd, dtype=np.float64)
    if spk_seq.ndim != 3:
        raise ValueError(f"spk must have shape (T,H,W), got {spk_seq.shape}")
    if lcd.shape[0] != spk_seq.shape[0]:
        raise ValueError(f"LCD length {lcd.shape[0]} does not match spk length {spk_seq.shape[0]}")

    length, height, width = spk_seq.shape
    spk_flat = spk_seq.reshape(length, -1)
    pixel_count = spk_flat.shape[1]

    x = np.arange(length, dtype=np.float64)
    dx = float(x[1] - x[0]) if len(x) >= 2 else 1.0
    x0 = float(x[0]) if len(x) else 0.0
    trap_pairs = 0.5 * (lcd[:-1] + lcd[1:]) * dx if len(lcd) >= 2 else np.array([], dtype=np.float64)
    cumsum_trap = np.cumsum(trap_pairs)

    all_intervals = []
    for pix_idx in range(pixel_count):
        spk_pos = np.where(spk_flat[:, pix_idx] > 0)[0]
        if spk_pos.size == 0:
            continue
        spk_pos = np.concatenate(([0], spk_pos))
        starts = spk_pos[:-1]
        ends = spk_pos[1:]
        valid = ends > starts
        if not np.any(valid):
            continue
        all_intervals.append(
            np.stack(
                (np.full_like(starts[valid], pix_idx), starts[valid], ends[valid]),
                axis=1,
            )
        )

    if not all_intervals:
        empty = np.empty((0, height, width), dtype=np.float32)
        return empty, empty

    intervals = np.vstack(all_intervals)
    start_idx = int(window_size)
    end_idx = length - int(window_size)
    if end_idx <= start_idx:
        empty = np.empty((0, height, width), dtype=np.float32)
        return empty, empty

    left_frames = []
    right_frames = []
    centers = range(start_idx, end_idx, int(window_size))
    iterator = tqdm(centers, desc="split spk", leave=False) if show_progress else centers
    interval_starts = intervals[:, 1]
    interval_ends = intervals[:, 2]

    for center in iterator:
        t1 = center - window_size
        t2 = center + window_size
        overlap = (interval_ends > t1) & (interval_starts < t2)
        window_intervals = intervals[overlap]

        if window_intervals.shape[0] == 0:
            left_frames.append(np.zeros((height, width), dtype=np.float32))
            right_frames.append(np.zeros((height, width), dtype=np.float32))
            continue

        segment_pix = window_intervals[:, 0].astype(np.int64)
        starts = window_intervals[:, 1].astype(np.float64)
        ends = window_intervals[:, 2].astype(np.float64)
        segment_dt = ends - starts

        if cumsum_trap.size > 0:
            t1_idx = np.round((starts - x0) / dx).astype(np.int64)
            t2_idx = np.round((ends - x0) / dx).astype(np.int64)
            max_idx = len(cumsum_trap)
            t1_idx = np.clip(t1_idx, 0, max_idx)
            t2_idx = np.clip(t2_idx, 0, max_idx)
            segment_i = np.zeros_like(segment_dt, dtype=np.float64)
            valid_int = t2_idx > t1_idx
            if np.any(valid_int):
                upper = cumsum_trap[t2_idx[valid_int] - 1]
                lower = np.zeros_like(upper, dtype=np.float64)
                valid_lower = t1_idx[valid_int] > 0
                lower[valid_lower] = cumsum_trap[t1_idx[valid_int][valid_lower] - 1]
                segment_i[valid_int] = upper - lower
        else:
            segment_i = np.zeros_like(segment_dt, dtype=np.float64)

        counts = np.zeros(pixel_count, dtype=np.int64)
        s_ii = np.zeros(pixel_count, dtype=np.float64)
        s_dd = np.zeros(pixel_count, dtype=np.float64)
        s_id = np.zeros(pixel_count, dtype=np.float64)
        s_ig = np.zeros(pixel_count, dtype=np.float64)
        s_dg = np.zeros(pixel_count, dtype=np.float64)

        np.add.at(counts, segment_pix, 1)
        np.add.at(s_ii, segment_pix, segment_i * segment_i)
        np.add.at(s_dd, segment_pix, segment_dt * segment_dt)
        np.add.at(s_id, segment_pix, segment_i * segment_dt)
        np.add.at(s_ig, segment_pix, segment_i)
        np.add.at(s_dg, segment_pix, segment_dt)

        lambda_reg = max(0.0, float(reg))
        det = (s_ii + lambda_reg) * (s_dd + lambda_reg) - s_id * s_id
        valid = (counts >= 1) & (np.abs(det) > 1e-12)

        right_flat = np.zeros(pixel_count, dtype=np.float64)
        left_flat = np.zeros(pixel_count, dtype=np.float64)
        det_valid = det[valid]
        right_flat[valid] = (
            s_ig[valid] * (s_dd[valid] + lambda_reg) - s_dg[valid] * s_id[valid]
        ) / det_valid
        left_flat[valid] = (
            (s_ii[valid] + lambda_reg) * s_dg[valid] - s_id[valid] * s_ig[valid]
        ) / det_valid

        left_frames.append(left_flat.reshape(height, width).astype(np.float32))
        right_frames.append(right_flat.reshape(height, width).astype(np.float32))

    left = np.stack(left_frames, axis=0) * float(threshold)
    right = np.stack(right_frames, axis=0) * float(threshold)
    return left.astype(np.float32), right.astype(np.float32)


@dataclass(frozen=True)
class SpikeInferenceConfig:
    npz_path: str
    sequence_length: int
    crop_size: tuple[int, int] | None
    max_files: int
    ref_source: str
    spike_reg: float
    split_progress: bool


class SpikeInferenceDataset(Dataset):
    """Dataset that builds model inputs from spk/LCD instead of stored left/right."""

    def __init__(self, cfg: SpikeInferenceConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.paths = collect_npz_paths(cfg.npz_path, max_files=cfg.max_files)
        print(f"inference dataset: {len(self.paths)} npz files found in {cfg.npz_path}")

    def __len__(self) -> int:
        return len(self.paths)

    def _clip_length(self, available_frames: int) -> int:
        if int(self.cfg.sequence_length) <= 0:
            return available_frames
        return max(1, min(int(self.cfg.sequence_length), available_frames))

    @staticmethod
    def _to_tensor(arr: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(arr.astype(np.float32)).unsqueeze(1)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        path = self.paths[idx]
        with np.load(path, mmap_mode="r") as data:
            spikes = np.asarray(data["spk"])
            lcd = np.asarray(data["LCD"])
            threshold = float(np.asarray(data["threshold"]).item())
            window_size = int(np.asarray(data["window_size"]).item()) if "window_size" in data.files else 83

            spikes = center_crop_array(spikes, self.cfg.crop_size)
            left_np, right_np = split_spikes_to_left_right(
                spikes,
                lcd,
                window_size=window_size,
                threshold=threshold,
                reg=float(self.cfg.spike_reg),
                show_progress=bool(self.cfg.split_progress),
            )

            if self.cfg.ref_source == "stored":
                stored_mt = load_stored_mt(data, self.cfg.crop_size)
                mt = (
                    stored_mt
                    if stored_mt is not None
                    else np.zeros_like(left_np, dtype=np.float32)
                )
            elif self.cfg.ref_source == "spk":
                mt = compute_mt_reference(
                    spikes,
                    window_size=window_size,
                    frame_count=left_np.shape[0],
                    reference_window_size=read_reference_window_size(data, window_size),
                )
            else:
                mt = np.zeros_like(left_np, dtype=np.float32)

            if "vid_left" in data.files:
                target_a = center_crop_array(np.asarray(data["vid_left"]), self.cfg.crop_size)
            else:
                target_a = np.zeros_like(left_np, dtype=np.float32)
            if "vid_right" in data.files:
                target_b = center_crop_array(np.asarray(data["vid_right"]), self.cfg.crop_size)
            else:
                target_b = np.zeros_like(right_np, dtype=np.float32)

        min_len = min(left_np.shape[0], right_np.shape[0], mt.shape[0], target_a.shape[0], target_b.shape[0])
        clip_len = self._clip_length(min_len)
        left_np = left_np[:clip_len]
        right_np = right_np[:clip_len]
        mt = mt[:clip_len]
        target_a = target_a[:clip_len]
        target_b = target_b[:clip_len]

        left = self._to_tensor(left_np)
        right = self._to_tensor(right_np)
        mt_t = self._to_tensor(mt)
        target_a_t = self._to_tensor(target_a)
        target_b_t = self._to_tensor(target_b)
        threshold_value = float(threshold)
        threshold_t = torch.tensor(threshold_value, dtype=torch.float32)

        return {
            "left": left.clamp(min=0.0, max=threshold_value),
            "right": right.clamp(min=0.0, max=threshold_value),
            "mt": mt_t,
            "target_a": target_a_t,
            "target_b": target_b_t,
            "f_clip": torch.zeros(clip_len, dtype=torch.float32),
            "intensity": torch.zeros_like(left),
            "threshold": threshold_t,
        }
