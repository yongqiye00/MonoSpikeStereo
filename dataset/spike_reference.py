"""Spike-derived M_t reference maps used by MonoSpikeStereo."""

from __future__ import annotations

import numpy as np


def compute_mt_reference(
    spikes: np.ndarray,
    window_size: int,
    *,
    frame_start: int = 0,
    frame_count: int | None = None,
    reference_window_size: int | None = None,
) -> np.ndarray:
    """Compute the paper-aligned mixed spike reference.

    The returned value is ``sum(spikes) / T_f`` for each output frame, where
    ``T_f`` is one modulation period. Callers multiply the result by the sensor
    threshold ``Q`` before feeding it to the model, matching
    ``M_t = Q / T_f * sum(1)``.
    """
    spikes = np.asarray(spikes)
    if spikes.ndim != 3:
        raise ValueError(f"spikes must have shape (T,H,W), got {spikes.shape}")

    window = int(window_size)
    if window <= 0:
        raise ValueError(f"window_size must be positive, got {window_size}")

    total_length, height, width = spikes.shape
    reference_window = int(reference_window_size or (4 * window))
    if reference_window <= 0:
        raise ValueError(
            f"reference_window_size must be positive, got {reference_window_size}"
        )

    frame_start = max(0, int(frame_start))
    max_count = max(0, ((total_length - window - 1) // window) - frame_start)
    if frame_count is None:
        frame_count = max_count
    else:
        frame_count = min(max(0, int(frame_count)), max_count)

    if frame_count == 0:
        return np.empty((0, height, width), dtype=np.float32)

    if reference_window % window != 0:
        return _compute_direct(
            spikes,
            window,
            reference_window,
            frame_start=frame_start,
            frame_count=frame_count,
        )

    period_blocks = max(1, reference_window // window)
    complete_blocks = total_length // window
    first_frame = frame_start
    last_frame = frame_start + frame_count - 1

    block_start = max(0, first_frame - period_blocks + 1)
    block_end = last_frame + 1
    if first_frame < period_blocks - 1:
        block_end = max(block_end, period_blocks)
    block_end = min(block_end, complete_blocks)

    span = spikes[block_start * window : block_end * window]
    block_count = block_end - block_start
    if block_count <= 0 or span.shape[0] != block_count * window:
        return _compute_direct(
            spikes,
            window,
            reference_window,
            frame_start=frame_start,
            frame_count=frame_count,
        )

    blocks = span.reshape(block_count, window, height, width)
    block_sums = blocks.sum(axis=1, dtype=np.float32)
    refs = np.zeros((frame_count, height, width), dtype=np.float32)

    for out_idx in range(frame_count):
        frame_idx = frame_start + out_idx
        if frame_idx < period_blocks - 1:
            start_block = 0
            end_block = period_blocks
        else:
            start_block = frame_idx - period_blocks + 1
            end_block = frame_idx + 1

        local_start = max(0, start_block - block_start)
        local_end = min(block_count, end_block - block_start)
        if local_end > local_start:
            refs[out_idx] = block_sums[local_start:local_end].sum(axis=0)

    refs /= float(reference_window)
    return refs


def _compute_direct(
    spikes: np.ndarray,
    window: int,
    reference_window: int,
    *,
    frame_start: int,
    frame_count: int,
) -> np.ndarray:
    total_length, height, width = spikes.shape
    refs = np.zeros((frame_count, height, width), dtype=np.float32)
    for out_idx in range(frame_count):
        frame_idx = frame_start + out_idx
        end = (frame_idx + 1) * window
        start = end - reference_window
        if start < 0:
            start = 0
            end = min(reference_window, total_length)
        else:
            end = min(end, total_length)
        if end > start:
            refs[out_idx] = spikes[start:end].sum(axis=0, dtype=np.float32)
    refs /= float(reference_window)
    return refs
