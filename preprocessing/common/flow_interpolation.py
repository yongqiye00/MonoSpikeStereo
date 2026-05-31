"""Flow-based frame interpolation helpers."""

from __future__ import annotations

import cv2
import numpy as np


def warp_image_backward(img: np.ndarray, flow_target_to_source: np.ndarray) -> np.ndarray:
    """Warp a source image by sampling source coordinates for each target pixel."""
    h, w = img.shape[:2]
    grid_y, grid_x = np.mgrid[0:h, 0:w].astype(np.float32)

    map_x = grid_x + flow_target_to_source[:, :, 0]
    map_y = grid_y + flow_target_to_source[:, :, 1]

    return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def compute_occlusion_mask(flow12: np.ndarray, flow21: np.ndarray, threshold: float = 1.0):
    """Compute simple visible masks using forward-backward flow consistency."""
    h, w = flow12.shape[:2]
    grid_y, grid_x = np.mgrid[0:h, 0:w].astype(np.float32)

    pts_x = grid_x + flow12[:, :, 0]
    pts_y = grid_y + flow12[:, :, 1]
    flow21_x = cv2.remap(
        flow21[:, :, 0].astype(np.float32),
        pts_x,
        pts_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    flow21_y = cv2.remap(
        flow21[:, :, 1].astype(np.float32),
        pts_x,
        pts_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    fb_dist = np.sqrt((flow12[:, :, 0] + flow21_x) ** 2 + (flow12[:, :, 1] + flow21_y) ** 2)
    mask1 = fb_dist < threshold

    pts2_x = grid_x + flow21[:, :, 0]
    pts2_y = grid_y + flow21[:, :, 1]
    flow12_x = cv2.remap(
        flow12[:, :, 0].astype(np.float32),
        pts2_x,
        pts2_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    flow12_y = cv2.remap(
        flow12[:, :, 1].astype(np.float32),
        pts2_x,
        pts2_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    fb2_dist = np.sqrt((flow21[:, :, 0] + flow12_x) ** 2 + (flow21[:, :, 1] + flow12_y) ** 2)
    mask2 = fb2_dist < threshold

    return mask1, mask2


def interpolate_frames_with_flow(
    img1: np.ndarray,
    img2: np.ndarray,
    flow_forward: np.ndarray,
    flow_backward: np.ndarray | None = None,
    num_interpolations: int = 4,
    clip_range: tuple[float, float] | None = (0, 255),
    bidirectional: bool = True,
):
    """Generate intermediate frames from two frames and precomputed optical flow."""
    interpolated_frames = []

    use_backward = bidirectional and flow_backward is not None
    img2_base = img2.astype(np.float32) if not use_backward else None
    for i in range(1, num_interpolations + 1):
        t = i / (num_interpolations + 1)
        flow_t_to_1 = -t * flow_forward
        warped_from_1 = warp_image_backward(img1, flow_t_to_1).astype(np.float32)

        if use_backward:
            flow_t_to_2 = -(1 - t) * flow_backward
            warped_from_2 = warp_image_backward(img2, flow_t_to_2).astype(np.float32)
        else:
            warped_from_2 = img2_base if img2_base is not None else img2.astype(np.float32)

        interpolated_frame = (1 - t) * warped_from_1 + t * warped_from_2
        if clip_range is not None:
            low, high = clip_range
            interpolated_frame = np.clip(interpolated_frame, low, high)
        interpolated_frames.append(interpolated_frame.astype(img1.dtype))

    return interpolated_frames
