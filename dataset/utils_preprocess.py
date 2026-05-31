import numpy as np
import torch

def match_brightness_moment(source, reference):
    """
    Adjust the intensity of the source image to match the mean and standard deviation
    of the reference image.

    Args:
        source: Input image array (T, H, W) or (H, W).
        reference: Reference image array (T, H, W) or (H, W).

    Returns:
        Matched source image.
    """
    if source.ndim == 3:
        matched = np.zeros_like(source)
        for i in range(source.shape[0]):
            matched[i] = _match_moment_single(source[i], reference[i])
        return matched
    else:
        return _match_moment_single(source, reference)

def _match_moment_single(source, reference):
    src_mean = np.mean(source)
    src_std = np.std(source)
    ref_mean = np.mean(reference)
    ref_std = np.std(reference)

    if src_std < 1e-6:
        return source # Avoid division by zero

    matched = (source - src_mean) * (ref_std / src_std) + ref_mean
    return np.clip(matched, 0, 1)

def temporal_smooth(sequence, alpha=0.5):
    """
    Apply an exponential moving average (EMA) for temporal smoothing.

    Args:
        sequence: (T, H, W) numpy array.
        alpha: Smoothing factor, 0 < alpha <= 1.
               Lower alpha means more smoothing (more history).
               Higher alpha means less smoothing (more current frame).

    Returns:
        Smoothed sequence.
    """
    if alpha >= 1.0:
        return sequence

    T, H, W = sequence.shape
    smoothed = np.zeros_like(sequence)

    smoothed[0] = sequence[0]
    for i in range(1, T):
        smoothed[i] = alpha * sequence[i] + (1 - alpha) * smoothed[i-1]

    return smoothed

def temporal_median_filter(sequence, window_size=3):
    """
    Apply a temporal median filter to remove transient noise/flicker.

    Args:
        sequence: (T, H, W) numpy array.
        window_size: Size of the window (odd number).

    Returns:
        Filtered sequence.
    """
    if window_size < 2:
        return sequence

    T, H, W = sequence.shape
    filtered = np.zeros_like(sequence)
    pad = window_size // 2

    # Pad sequence
    padded = np.pad(sequence, ((pad, pad), (0, 0), (0, 0)), mode='edge')

    for i in range(T):
        window = padded[i:i+window_size]
        filtered[i] = np.median(window, axis=0)

    return filtered

class TemporalSmoother:
    def __init__(self, window_size=5, method='median'):
        """
        Stateful temporal smoother for sequential processing.

        Args:
            window_size: Number of frames to smooth over.
            method: 'mean' or 'median'.
        """
        self.window_size = window_size
        self.method = method
        self.buffer = []

    def update(self, frame):
        """
        Add a new frame and return the smoothed result.

        Args:
            frame: torch.Tensor or np.ndarray (H, W) or (C, H, W)

        Returns:
            Smoothed frame of the same type and shape.
        """
        # Detach if tensor to avoid graph growth if keeping history
        if isinstance(frame, torch.Tensor):
            frame_store = frame.detach()
        else:
            frame_store = frame

        self.buffer.append(frame_store)
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)

        if len(self.buffer) == 0:
            return frame

        if isinstance(frame, torch.Tensor):
            stack = torch.stack(self.buffer, dim=0)
            if self.method == 'mean':
                return torch.mean(stack, dim=0)
            elif self.method == 'median':
                return torch.median(stack, dim=0).values
        else:
            stack = np.array(self.buffer)
            if self.method == 'mean':
                return np.mean(stack, axis=0)
            elif self.method == 'median':
                return np.median(stack, axis=0)
        return frame

    def reset(self):
        self.buffer = []
