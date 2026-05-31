import atexit
import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image


def _to_uint8_img(t: torch.Tensor) -> np.ndarray:
    """
    Convert a tensor to an RGB uint8 image.

    Accepts (B,C,H,W), (C,H,W), or (H,W) tensors with values in [0,1] or
    [0,255].
    """
    if isinstance(t, torch.Tensor):
        t = t.detach().cpu()
    if t.ndim == 4:
        t = t[0]
    if t.ndim == 3:
        if t.shape[0] == 1:
            t = t.repeat(3, 1, 1)
        t = t.permute(1, 2, 0)
    elif t.ndim == 2:
        t = t.unsqueeze(-1).repeat(1, 1, 3)

    arr = t.numpy()
    if arr.max() <= 1.0:
        arr = (arr * 255.0).clip(0, 255)
    return arr.astype(np.uint8)


@dataclass
class SaveConfig:
    root_dir: str = "./outputs_eval"
    make_grid: bool = False
    export_video: bool = True
    video_fps: int = 12


class OutputSaver:
    def __init__(self, cfg: Optional[SaveConfig] = None):
        self.cfg = cfg or SaveConfig()
        self._video_writers: Dict[str, cv2.VideoWriter] = {}
        atexit.register(self.close)

    def _dir(self, method: str, sample_idx: int) -> str:
        d = os.path.join(self.cfg.root_dir, method, f"sample_{sample_idx:03d}")
        os.makedirs(d, exist_ok=True)
        return d

    def _frame_dir(self, method: str, sample_idx: int, frame_idx: int) -> str:
        d = os.path.join(self._dir(method, sample_idx), f"frame_{frame_idx:03d}")
        os.makedirs(d, exist_ok=True)
        return d

    def _videos_dir(self, method: str, sample_idx: int) -> str:
        d = os.path.join(self._dir(method, sample_idx), "videos")
        os.makedirs(d, exist_ok=True)
        return d

    def _video_key(self, path: str) -> str:
        return os.path.abspath(path)

    def _ensure_bgr(self, img: np.ndarray) -> np.ndarray:
        if img.ndim == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    def _get_video_writer(self, video_path: str, frame_size: Tuple[int, int]) -> cv2.VideoWriter:
        key = self._video_key(video_path)
        writer = self._video_writers.get(key)
        if writer is not None:
            return writer

        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(video_path, fourcc, float(self.cfg.video_fps), frame_size)
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open video writer for: {video_path}")
        self._video_writers[key] = writer
        return writer

    def _append_video_frame(
        self,
        *,
        method: str,
        sample_idx: int,
        tag: str,
        img: np.ndarray,
    ) -> None:
        if not self.cfg.export_video:
            return
        h, w = img.shape[:2]
        video_path = os.path.join(self._videos_dir(method, sample_idx), f"{tag}.mp4")
        writer = self._get_video_writer(video_path, (w, h))
        writer.write(self._ensure_bgr(img))

    def save_rgb(
        self,
        tensor: torch.Tensor,
        path: str,
        *,
        method: Optional[str] = None,
        sample_idx: Optional[int] = None,
        tag: Optional[str] = None,
    ) -> None:
        img = _to_uint8_img(tensor)
        Image.fromarray(img).save(path)
        if method is not None and sample_idx is not None and tag is not None:
            self._append_video_frame(method=method, sample_idx=sample_idx, tag=tag, img=img)

    def close_sample_videos(self, method: str, sample_idx: int) -> None:
        sample_videos_dir = os.path.abspath(
            os.path.join(self.cfg.root_dir, method, f"sample_{sample_idx:03d}", "videos")
        )
        sample_prefix = sample_videos_dir + os.sep

        for key, writer in list(self._video_writers.items()):
            if key == sample_videos_dir or key.startswith(sample_prefix):
                try:
                    writer.release()
                except (cv2.error, RuntimeError, OSError) as exc:
                    print(f"[WARN] Failed to release video writer {key}: {exc}")
                self._video_writers.pop(key, None)

    def close(self) -> None:
        for key, writer in self._video_writers.items():
            try:
                writer.release()
            except (cv2.error, RuntimeError, OSError) as exc:
                print(f"[WARN] Failed to release video writer {key}: {exc}")
        self._video_writers.clear()

    def __del__(self):
        self.close()

    def save_sample_frame(
        self,
        method: str,
        sample_idx: int,
        frame_idx: int,
        tensors: Dict[str, torch.Tensor],
    ) -> None:
        frame_dir = self._frame_dir(method, sample_idx, frame_idx)
        image_keys = (
            "left_input",
            "right_input",
            "mixed_input",
            "mt",
            "left_pred",
            "right_pred",
            "left_gt",
            "right_gt",
        )

        for key in image_keys:
            if key not in tensors:
                continue
            self.save_rgb(
                tensors[key],
                os.path.join(frame_dir, f"{key}.png"),
                method=method,
                sample_idx=sample_idx,
                tag=key,
            )


__all__ = ["OutputSaver", "SaveConfig"]
