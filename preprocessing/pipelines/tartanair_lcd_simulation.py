import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from common.simulation_utils import (
    ensure_dir,
    nhwc_to_nhw_gray,
)
from utils.LCD_wave import create_waveform
from utils.simu_utils import Spk_SlidingWindow_xy_hw_optimized
from utils.video_reader import VideoReader
from pipelines.tartanair_lcd_helpers import (
    compute_mt_reference_from_spikes,
    simulate_spikes_to_memmap,
)


@dataclass(frozen=True)
class TartanAirLCDRuntime:
    crop_size: Optional[Tuple[int, int]]
    save_spike_sequences: bool
    left_intensity_scale: float = 1.0
    lcd_max_value: float = 0.5
    extension_length: int = 20
    waveform_period: int = 332
    window_size: int = 83


def run_simulation_LCD(root_left: str, root_right: str, out_dir: str, crop_size: Optional[Tuple[int, int]],
                       random_crop: bool, threshold: float, length: int, use_noise: bool, noise_level: float,
                       save_spike_sequences: bool = True):
    # Shared CLI values; the LCD path samples the per-sequence values below.
    _ = random_crop, threshold, length, use_noise, noise_level
    runtime = TartanAirLCDRuntime(
        crop_size=crop_size,
        save_spike_sequences=save_spike_sequences,
    )

    vr_left = VideoReader(root_left, crop_size=runtime.crop_size, random_crop=False)
    vr_right = VideoReader(root_right, crop_size=runtime.crop_size, random_crop=False)
    if len(vr_left) != len(vr_right):
        raise ValueError(f"left/right sequence count mismatch: {len(vr_left)} vs {len(vr_right)}")

    ensure_dir(out_dir)

    for i in range(len(vr_left)):
        scene_name = vr_left.scenes[i]
        out_path = os.path.join(out_dir, f'{scene_name}.npz')

        print(f'Processing scene {i}/{len(vr_left)-1}: {scene_name}')
        vid_left = vr_left[i]  # NHWC
        vid_right = vr_right[i]  # NHWC

        vid_left_gray = nhwc_to_nhw_gray(vid_left).astype(np.float32) / 255.0
        vid_right_gray = nhwc_to_nhw_gray(vid_right).astype(np.float32) / 255.0
        total_length = vid_left_gray.shape[0] * runtime.extension_length

        print('input shape: ', vid_left_gray.shape)

        LCD = create_waveform(
            period=runtime.waveform_period,
            max_value=runtime.lcd_max_value,
            min_value=0.0,
            length=total_length,
        )[1]
        x = np.arange(total_length)
        y = LCD

        sampled_noise_level = np.random.uniform(0.005, 0.1)
        sampled_threshold = np.random.randint(6, 12)

        spikes_path, spikes = simulate_spikes_to_memmap(
            vid_left_gray=vid_left_gray,
            vid_right_gray=vid_right_gray,
            lcd=LCD,
            threshold=sampled_threshold,
            noise_level=sampled_noise_level,
            left_intensity_scale=runtime.left_intensity_scale,
            exten_length=runtime.extension_length,
            out_dir=out_dir,
            scene_name=scene_name,
        )

        right, left = Spk_SlidingWindow_xy_hw_optimized(
            spikes,
            x=x,
            y=y,
            window_size=runtime.window_size,
        )
        right = right * sampled_threshold
        left = left * sampled_threshold

        mt = compute_mt_reference_from_spikes(
            spikes,
            window_size=runtime.window_size,
            frame_count=left.shape[0],
            reference_window_size=runtime.waveform_period,
        )

        sample_indices = np.arange(total_length)[::runtime.window_size][1:-1]
        source_indices = np.clip(
            sample_indices // runtime.extension_length,
            0,
            vid_left_gray.shape[0] - 1,
        )

        save_payload = {
            'window_size': runtime.window_size,
            'threshold': sampled_threshold,
            'LCD': LCD,
            'left': left,
            'right': right,
            'mt': mt,
            'vid_left': vid_left_gray[source_indices],
            'vid_right': vid_right_gray[source_indices],
        }
        if runtime.save_spike_sequences:
            save_payload['spk'] = spikes

        print('out_path: ', out_path)
        np.savez_compressed(out_path, **save_payload)

        del spikes
        if os.path.exists(spikes_path):
            os.remove(spikes_path)
