"""Least-squares spike decoupling utilities used by preprocessing."""

from __future__ import annotations

import numpy as np
from tqdm import tqdm


def _prepare_spikes(spk_seq: np.ndarray) -> tuple[np.ndarray, int, int, int]:
    spk_seq = np.asarray(spk_seq)
    if spk_seq.ndim == 3:
        length, height, width = spk_seq.shape
        return spk_seq, length, height, width
    if spk_seq.ndim == 2:
        length, height = spk_seq.shape
        return spk_seq.reshape(length, height, 1), length, height, 1
    if spk_seq.ndim == 1:
        length = spk_seq.shape[0]
        return spk_seq.reshape(length, 1, 1), length, 1, 1
    raise ValueError(f"spk_seq must be 1D, 2D, or 3D, got shape {spk_seq.shape}")


def _integral_prefix(x: np.ndarray, y: np.ndarray) -> tuple[float, float, np.ndarray]:
    dx = float(x[1] - x[0]) if len(x) >= 2 else 1.0
    x0 = float(x[0]) if len(x) > 0 else 0.0
    if len(y) < 2:
        return dx, x0, np.array([], dtype=float)
    trap_pairs = 0.5 * (y[:-1] + y[1:]) * dx
    return dx, x0, np.cumsum(trap_pairs)


def _collect_spike_intervals(spk_flat: np.ndarray) -> np.ndarray:
    intervals = []
    for pix_idx in range(spk_flat.shape[1]):
        spike_pos = np.where(spk_flat[:, pix_idx] > 0)[0]
        if spike_pos.size == 0:
            continue

        spike_pos = np.concatenate(([0], spike_pos))
        starts = spike_pos[:-1]
        ends = spike_pos[1:]
        valid = (ends - starts) >= 0
        if not np.any(valid):
            continue

        intervals.append(
            np.stack(
                (
                    np.full_like(starts[valid], pix_idx),
                    starts[valid],
                    ends[valid],
                ),
                axis=1,
            )
        )

    if not intervals:
        return np.empty((0, 3), dtype=np.int64)
    return np.vstack(intervals)


def _integrate_segments(
    starts: np.ndarray,
    ends: np.ndarray,
    *,
    dx: float,
    x0: float,
    cumsum_trap: np.ndarray,
) -> np.ndarray:
    if cumsum_trap.size == 0:
        return np.zeros_like(starts, dtype=float)

    t1_idx = np.round((starts - x0) / dx).astype(int)
    t2_idx = np.round((ends - x0) / dx).astype(int)
    max_idx = len(cumsum_trap)
    t1_idx = np.clip(t1_idx, 0, max_idx)
    t2_idx = np.clip(t2_idx, 0, max_idx)

    segment_i = np.zeros_like(starts, dtype=float)
    valid = t2_idx > t1_idx
    if np.any(valid):
        upper = cumsum_trap[t2_idx[valid] - 1]
        lower = np.zeros_like(upper, dtype=float)
        has_lower = t1_idx[valid] > 0
        lower[has_lower] = cumsum_trap[t1_idx[valid][has_lower] - 1]
        segment_i[valid] = upper - lower
    return segment_i


def _solve_sliding_windows(
    spk_seq,
    x,
    y,
    *,
    window_size: int,
    step_size: int,
    reg: float,
    max_val: float,
):
    spk_seq, length, height, width = _prepare_spikes(np.asarray(spk_seq))
    x = np.asarray(x)
    y = np.asarray(y)

    spk_flat = spk_seq.reshape(length, -1)
    pixel_count = spk_flat.shape[1]
    dx, x0, cumsum_trap = _integral_prefix(x, y)
    all_intervals = _collect_spike_intervals(spk_flat)
    if all_intervals.size == 0:
        return np.array([]).reshape(0, height, width), np.array([]).reshape(0, height, width)

    start_idx = int(window_size)
    end_idx = length - int(window_size)
    if end_idx <= start_idx:
        return np.array([]).reshape(0, height, width), np.array([]).reshape(0, height, width)

    res_a = []
    res_b = []
    centers = range(start_idx, end_idx, max(1, int(step_size)))
    for center in tqdm(centers, desc="split spikes", leave=False):
        t1 = center - window_size
        t2 = center + window_size

        starts_all = all_intervals[:, 1]
        ends_all = all_intervals[:, 2]
        window_intervals = all_intervals[(ends_all > t1) & (starts_all < t2)]

        if window_intervals.shape[0] == 0:
            res_a.append(np.zeros((height, width)))
            res_b.append(np.zeros((height, width)))
            continue

        window_intervals = window_intervals[window_intervals[:, 0].argsort()]
        pix = window_intervals[:, 0].astype(int)
        starts = window_intervals[:, 1]
        ends = window_intervals[:, 2]
        segment_dt = (ends - starts).astype(float) * max_val
        segment_i = _integrate_segments(
            starts,
            ends,
            dx=dx,
            x0=x0,
            cumsum_trap=cumsum_trap,
        )
        segment_g = np.ones_like(segment_dt, dtype=float)

        counts = np.zeros(pixel_count, dtype=int)
        np.add.at(counts, pix, 1)

        s_ii = np.zeros(pixel_count, dtype=float)
        s_dd = np.zeros(pixel_count, dtype=float)
        s_id = np.zeros(pixel_count, dtype=float)
        s_ig = np.zeros(pixel_count, dtype=float)
        s_dg = np.zeros(pixel_count, dtype=float)
        np.add.at(s_ii, pix, segment_i * segment_i)
        np.add.at(s_dd, pix, segment_dt * segment_dt)
        np.add.at(s_id, pix, segment_i * segment_dt)
        np.add.at(s_ig, pix, segment_i * segment_g)
        np.add.at(s_dg, pix, segment_dt * segment_g)

        lambda_reg = max(0.0, float(reg)) if reg is not None else 0.0
        det = (s_ii + lambda_reg) * (s_dd + lambda_reg) - s_id * s_id
        valid = (counts >= 1) & (np.abs(det) > 1e-12)

        a_flat = np.zeros(pixel_count, dtype=float)
        b_flat = np.zeros(pixel_count, dtype=float)
        det_valid = det[valid]
        a_flat[valid] = (s_ig[valid] * (s_dd[valid] + lambda_reg) - s_dg[valid] * s_id[valid]) / det_valid
        b_flat[valid] = ((s_ii[valid] + lambda_reg) * s_dg[valid] - s_id[valid] * s_ig[valid]) / det_valid

        res_a.append(a_flat.reshape(height, width))
        res_b.append(b_flat.reshape(height, width))

    return np.array(res_a), np.array(res_b)


def Spk_SlidingWindow_xy_hw_optimized(
    spk_seq,
    x,
    y,
    window_size: int = 20,
    reg: float = 1e-3,
):
    """Estimate the two stereo components from mixed spikes with sliding LS."""
    return _solve_sliding_windows(
        spk_seq,
        x,
        y,
        window_size=int(window_size),
        step_size=int(window_size),
        reg=reg,
        max_val=1.0,
    )


def Spk_SlidingWindow_xy_hw_optimized_with_step(
    spk_seq,
    x,
    y,
    window_size: int = 20,
    step_size: int | None = None,
    reg: float = 1e-3,
    max_val: float = 1.0,
):
    """Variant of ``Spk_SlidingWindow_xy_hw_optimized`` with a custom stride."""
    return _solve_sliding_windows(
        spk_seq,
        x,
        y,
        window_size=int(window_size),
        step_size=int(step_size or max(1, window_size // 2)),
        reg=reg,
        max_val=max_val,
    )
