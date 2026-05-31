"""LCD modulation waveform helpers."""

from __future__ import annotations

import numpy as np


def create_waveform(period: int, max_value: float = 1.0, min_value: float = 0.0, length: int | None = None):
    """Create a cosine LCD attenuation waveform."""
    phase = np.arange(period) + np.random.randint(0, period)
    y = np.cos(2 * np.pi * phase / period) + 1.0
    y = y * ((max_value - min_value) / 2.0) + min_value

    if length is not None:
        repeats = length // period + 1
        return np.arange(length), np.tile(y, repeats)[:length]
    return phase, y


def create_waveform_random(
    period: int,
    max_value: float = 1.0,
    min_value: float = 0.0,
    length: int | None = None,
    noise_level: float = 0.05,
):
    """Create a monotonic-down-then-up random LCD attenuation waveform."""
    base_half = period // 2
    jitter_max = max(1, int(base_half * 0.2))
    half = int(np.clip(base_half + np.random.randint(-jitter_max, jitter_max + 1), 1, period - 1))
    amplitude = max_value - min_value

    dec = np.linspace(max_value, min_value, half)
    dec += np.random.randn(half) * (noise_level * amplitude)
    for idx in range(1, half):
        dec[idx] = min(dec[idx], dec[idx - 1])

    inc_len = period - half
    inc = np.linspace(min_value, max_value, inc_len)
    inc += np.random.randn(inc_len) * (noise_level * amplitude)
    for idx in range(1, inc_len):
        inc[idx] = max(inc[idx], inc[idx - 1])

    y = np.clip(np.concatenate([dec, inc], axis=0), min_value, max_value)
    if length is None:
        return np.arange(period), y

    repeats = int(np.ceil(length / period))
    return np.arange(length), np.tile(y, repeats)[:length]
