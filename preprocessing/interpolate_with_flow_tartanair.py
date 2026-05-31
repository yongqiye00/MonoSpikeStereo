import argparse
import os
import re
from dataclasses import dataclass
from glob import glob

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import warnings

from runtime import (
    bootstrap_interpolation_runtime,
    load_cli_config,
    parse_hw_size,
    resolve_raft_checkpoint,
)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

bootstrap_interpolation_runtime()

from common.flow_interpolation import interpolate_frames_with_flow
from common.raft_flow import compute_optical_flow, load_raft_model


@dataclass(frozen=True)
class InterpolationRuntime:
    model: object
    device: str
    output_root: str
    crop_size: object
    num_interpolations: int
    max_frames: int | None
    group_size: int
    max_groups: int
    group_skip: int
    use_bidirectional_flow: bool
    verbose: bool = False


@dataclass(frozen=True)
class TartanAirView:
    scene: str
    kind: str
    p_folder: str
    p_path: str
    view: str
    view_path: str


def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def log_verbose(runtime: InterpolationRuntime, *args):
    if runtime.verbose:
        print(*args)


def save_image(arr, path):
    """Save an image array to disk."""
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 3 and arr.shape[2] == 3:
        Image.fromarray(arr, 'RGB').save(path)
    else:
        Image.fromarray(arr).save(path)

def apply_crop(img, crop_size, random_crop=False):
    """Apply center or random crop. img is HWC numpy array."""
    if crop_size is None:
        return img
    h, w = img.shape[:2]
    ch, cw = crop_size
    if h < ch or w < cw:
        # if image smaller than crop, resize by padding or center crop fallback
        return img
    if random_crop:
        top = np.random.randint(0, h - ch + 1)
        left = np.random.randint(0, w - cw + 1)
    else:
        top = (h - ch) // 2
        left = (w - cw) // 2
    return img[top:top + ch, left:left + cw]


def natural_sort_key(value):
    name = os.path.basename(str(value))
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r'(\d+)', name)]


def list_subdirs(path):
    return [d for d in sorted(os.listdir(path)) if os.path.isdir(os.path.join(path, d))]


def load_image_group(img_paths, crop_size):
    imgs = [np.array(Image.open(p).convert('RGB')) for p in img_paths]
    return [apply_crop(im, crop_size, random_crop=False) for im in imgs]


def save_frame_group(output_dir, frames):
    ensure_dir(output_dir)
    saw_existing = False
    for idx, frame in enumerate(frames):
        out_name = f"{idx:06d}.png"
        out_path = os.path.join(output_dir, out_name)
        if os.path.exists(out_path):
            print(f"Warning: output frame {out_path} already exists, skipping.")
            saw_existing = True
            continue
        save_image(frame, out_path)
    return saw_existing


def process_view_groups(runtime: InterpolationRuntime, view_ctx: TartanAirView):
    log_verbose(runtime, 'view_path', view_ctx.view_path)
    images = sorted(glob(os.path.join(view_ctx.view_path, '*.png')), key=natural_sort_key)
    if runtime.max_frames is not None and runtime.max_frames > 0:
        images = images[:runtime.max_frames]
    max_groups = max(1, int(runtime.max_groups))

    skip_flag = False
    for im_idx in range(0, len(images)-1, runtime.group_skip):
        if skip_flag:
            break
        if im_idx//runtime.group_skip >= max_groups:
            break

        img_paths = images[im_idx: im_idx + runtime.group_size]
        if len(img_paths) < 2:
            continue
        imgs = load_image_group(img_paths, runtime.crop_size)

        output = process_scene(
            runtime.model,
            np.stack(imgs, axis=0),
            runtime.device,
            num_interpolations=runtime.num_interpolations,
            use_bidirectional_flow=runtime.use_bidirectional_flow,
        )

        frames = output['all_frames']

        group_name = f'group_{im_idx//runtime.group_skip:04d}'
        output_dir = os.path.join(
            runtime.output_root,
            view_ctx.scene,
            view_ctx.kind,
            view_ctx.p_folder,
            view_ctx.view,
            group_name,
        )
        if save_frame_group(output_dir, frames):
            skip_flag = True


def process_scene(
    model,
    imgs,
    device,
    num_interpolations=4,
    use_bidirectional_flow=False,
):
    """Interpolate all frames in one image group."""
    N = imgs.shape[0]
    all_frames = []

    for i in tqdm(range(N - 1), desc="Processing frames"):
        img1 = imgs[i]
        img2 = imgs[i + 1]

        all_frames.append(img1)

        flow_forward = compute_optical_flow(model, img1, img2, device)
        flow_backward = None
        if use_bidirectional_flow:
            flow_backward = compute_optical_flow(model, img2, img1, device)

        interpolated = interpolate_frames_with_flow(
            img1,
            img2,
            flow_forward,
            flow_backward,
            num_interpolations=num_interpolations,
            clip_range=(0, 255),
            bidirectional=use_bidirectional_flow,
        )

        all_frames.extend(interpolated)

    all_frames.append(imgs[-1])
    return {'all_frames': all_frames}

def main(
    input_root,
    output_root,
    raft_model_path,
    num_interpolations=4,
    crop_size=None,
    max_frames=20,
    group_size=60,
    max_groups=100,
    group_skip=1,
    use_bidirectional_flow=True,
    verbose=False,
):
    """Run RAFT-based interpolation for TartanAir-style stereo folders."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    ensure_dir(output_root)

    group_skip = max(1, int(group_skip))
    use_bidirectional_flow = bool(use_bidirectional_flow)

    print("Loading RAFT model...")
    model = load_raft_model(raft_model_path, device)
    runtime = InterpolationRuntime(
        model=model,
        device=device,
        output_root=output_root,
        crop_size=crop_size,
        num_interpolations=num_interpolations,
        max_frames=max_frames,
        group_size=group_size,
        max_groups=max_groups,
        group_skip=group_skip,
        use_bidirectional_flow=use_bidirectional_flow,
        verbose=bool(verbose),
    )

    scenes = list_subdirs(input_root)
    print(f"Found scenes: {scenes}")

    for scene in tqdm(scenes):

        scene_path = os.path.join(input_root, scene)
        kinds = list_subdirs(scene_path)

        for kind in kinds:

            kind_path = os.path.join(scene_path, kind)
            p_folders = list_subdirs(kind_path)
            print(f"Processing kind '{kind}' with {len(p_folders)} folders. {p_folders}")
            for p_folder in p_folders:
                p_path = os.path.join(kind_path, p_folder)

                views = list_subdirs(p_path)
                if 'image_left' not in views or 'image_right' not in views:
                    print(f"Skipping {p_path} because image_left/image_right are missing.")
                    continue
                for view in views:

                    view_path = os.path.join(p_path, view)

                    if 'depth' in view.lower():
                        continue
                    view_ctx = TartanAirView(
                        scene=scene,
                        kind=kind,
                        p_folder=p_folder,
                        p_path=p_path,
                        view=view,
                        view_path=view_path,
                    )
                    process_view_groups(runtime, view_ctx)


def parse_args(argv=None):
    cfg, pre_args = load_cli_config(argv, default_config="tartanair_interpolate.yaml")

    parser = argparse.ArgumentParser(description='Video frame interpolation using optical flow')
    parser.add_argument('--config', type=str, default=pre_args.config,
                       help='YAML config path or preset name')
    parser.add_argument('--input-root', type=str,
                       help='Root directory of source TartanAir frames.', default='data/tartanair/raw')
    parser.add_argument('--output-root', type=str,
                       help='Output directory for interpolated frames.', default='data/tartanair/interp')
    parser.add_argument('--raft-model-path', '--raft-model', dest='raft_model_path', type=str,
                       help='Path to the RAFT checkpoint.', default=None)
    parser.add_argument('--num-interpolations', type=int, default=59,
                       help='Number of frames inserted between each source-frame pair.')
    parser.add_argument('--crop-size', type=str, default='320x320',
                       help='Center crop size in HxW format, e.g. 512x512.')
    parser.add_argument('--max-frames', type=int, default=None,
                       help='Optional cap on source frames per sequence.')
    parser.add_argument('--group-size', type=int, default=3,
                       help='Number of source frames per interpolation group.')
    parser.add_argument('--max-groups', type=int, default=100,
                       help='Maximum number of groups to process per sequence.')
    parser.add_argument('--group-skip', type=int, default=4,
                       help='Stride between neighboring groups.')
    parser.add_argument('--single-flow', action='store_true',
                       help='Use only forward optical flow.')
    parser.add_argument('--verbose', action='store_true',
                       help='Print per-view and per-group diagnostics.')

    parser.set_defaults(**{
        key: value
        for key, value in cfg.items()
        if key != 'config'
    })

    args = parser.parse_args(argv)

    if getattr(args, 'use_bidirectional_flow', True) is False:
        args.single_flow = True

    return args


if __name__ == '__main__':
    args = parse_args()

    try:
        crop_size = parse_hw_size(args.crop_size)
    except ValueError as exc:
        print(f"Invalid crop size format: {args.crop_size}")
        raise SystemExit(1) from exc

    raft_model_path = resolve_raft_checkpoint(args.raft_model_path)

    main(
        args.input_root,
        args.output_root,
        raft_model_path,
        args.num_interpolations,
        crop_size,
        args.max_frames,
        args.group_size,
        max_groups=args.max_groups,
        use_bidirectional_flow=not args.single_flow,
        group_skip=args.group_skip,
        verbose=args.verbose,
    )
