import os

import numpy as np

from common.simulation_utils import ensure_dir


def simulate_spikes_to_memmap(
    vid_left_gray: np.ndarray,
    vid_right_gray: np.ndarray,
    lcd: np.ndarray,
    threshold: float,
    noise_level: float,
    left_intensity_scale: float,
    exten_length: int,
    out_dir: str,
    scene_name: str,
):
    """Generate spikes without materializing the expanded intensity volume."""
    n_frames, h, w = vid_left_gray.shape
    total_length = lcd.shape[0]
    spikes_path = os.path.join(out_dir, f".{scene_name}_spikes.npy")
    ensure_dir(out_dir)
    spikes = np.lib.format.open_memmap(
        spikes_path,
        mode="w+",
        dtype=np.uint8,
        shape=(total_length, h, w),
    )

    accumulated = np.zeros((h, w), dtype=np.float32)
    noise = np.random.normal(loc=0.0, scale=noise_level, size=(h, w)).astype(np.float32)
    noise = np.clip(noise, 0, 0.01)

    out_idx = 0
    for frame_idx in range(n_frames):
        left_frame = vid_left_gray[frame_idx] * left_intensity_scale
        right_frame = vid_right_gray[frame_idx]
        for _ in range(exten_length):
            current_frame = left_frame + right_frame * lcd[out_idx]
            accumulated += current_frame + noise
            spike_positions = accumulated >= threshold
            spikes[out_idx][spike_positions] = 1
            accumulated[spike_positions] -= threshold
            if np.any(accumulated < 0):
                np.clip(accumulated, 0.0, None, out=accumulated)
            out_idx += 1

    spikes.flush()
    return spikes_path, spikes


def compute_mt_reference_from_spikes(
    spikes: np.ndarray,
    window_size: int,
    frame_count: int,
    reference_window_size: int | None = None,
) -> np.ndarray:
    """Compute ``sum(spikes) / T_f`` M_t references aligned with model frames."""
    total_length, h, w = spikes.shape
    if total_length == 0 or frame_count <= 0:
        return np.empty((0, h, w), dtype=np.float32)

    window = int(window_size)
    reference_window = int(reference_window_size or (4 * window))
    complete_blocks = total_length // window
    if complete_blocks <= 0:
        return np.empty((0, h, w), dtype=np.float32)

    frame_count = min(int(frame_count), max(0, ((total_length - window - 1) // window)))
    if frame_count <= 0:
        return np.empty((0, h, w), dtype=np.float32)

    refs = np.zeros((frame_count, h, w), dtype=np.float32)
    period_blocks = max(1, reference_window // window)
    trimmed = spikes[: complete_blocks * window]
    block_sums = trimmed.reshape(complete_blocks, window, h, w).sum(axis=1, dtype=np.float32)

    for frame_idx in range(frame_count):
        if frame_idx < period_blocks - 1:
            start_block = 0
            end_block = min(period_blocks, complete_blocks)
        else:
            start_block = frame_idx - period_blocks + 1
            end_block = frame_idx + 1

        if end_block > start_block:
            refs[frame_idx] = block_sums[start_block:end_block].sum(axis=0)

    refs /= float(reference_window)
    return refs
