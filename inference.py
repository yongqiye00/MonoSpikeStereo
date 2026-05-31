"""Inference from spike sequences for trained MonoSpikeStereo checkpoints.

Unlike test.py, this entrypoint does not consume precomputed ``left`` and
``right`` arrays from the NPZ payload. It starts from ``spk`` + ``LCD``,
reconstructs the left/right observations with the same sliding-window splitter
used during preprocessing, and then feeds those observations into the network.
"""

from __future__ import annotations

import argparse
import os
import tempfile
from collections.abc import Iterable
from contextlib import nullcontext

os.environ.setdefault(
    "MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib-monospikestereo")
)

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.spike_inference import SpikeInferenceConfig, SpikeInferenceDataset
from src.train.setup import build_model, make_device, maybe_compile_model
from utils.argparse_utils import add_bool_arg
from utils.dataloader_utils import build_dataloader
from utils.entrypoint_utils import parse_train_args_with_extra_options, resolve_checkpoint
from utils.eval_utils import (
    load_model_checkpoint,
    save_frame_outputs,
)
from utils.output_saver import OutputSaver, SaveConfig
from utils.sequence_utils import (
    StateEntry,
    append_and_truncate,
    prepare_inputs,
    update_history,
)
from utils.utils_train import parse_crop_size


def _extra_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        add_help=False,
        description="Run MonoSpikeStereo inference from spk/LCD NPZ data.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Explicit checkpoint path. Overrides --resume/--resume_path and best checkpoint lookup.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Root directory for saved inference images.",
    )
    parser.add_argument(
        "--save_method_name",
        type=str,
        default="inference",
        help="Subdirectory name used by OutputSaver.",
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=0,
        help="Limit number of NPZ files to process (<=0 uses all files).",
    )
    parser.add_argument(
        "--save_every_n_frames",
        type=int,
        default=1,
        help="Save predictions every N inferred frames.",
    )
    parser.add_argument(
        "--ref_source",
        choices=("stored", "spk", "zeros"),
        default="spk",
        help=(
            "Reference input source. 'spk' computes the paper-aligned M_t map "
            "from raw spikes; 'stored' uses a stored NPZ reference when present; "
            "'zeros' uses a zero reference."
        ),
    )
    parser.add_argument(
        "--spike_reg",
        type=float,
        default=1e-3,
        help="Regularization used by the spike left/right splitter.",
    )
    add_bool_arg(
        parser,
        "save_inputs",
        default=False,
        help_text="Save reconstructed left/right observation images.",
    )
    add_bool_arg(
        parser,
        "save_gt",
        default=False,
        help_text="Save vid_left/vid_right targets when they exist in the NPZ.",
    )
    add_bool_arg(
        parser,
        "export_videos",
        default=True,
        help_text="Also export per-sample MP4 videos.",
    )
    add_bool_arg(
        parser,
        "split_progress",
        default=False,
        help_text="Show per-window progress while splitting spikes.",
    )
    return parser


def parse_inference_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    extra = _extra_parser()
    return parse_train_args_with_extra_options(
        argv,
        extra_parser=extra,
        options_title="Inference options",
        default_output_root="inference_outputs",
    )


def _build_loader(args: argparse.Namespace) -> DataLoader:
    dataset = SpikeInferenceDataset(
        SpikeInferenceConfig(
            npz_path=args.npz_path,
            sequence_length=int(args.sequence_length),
            crop_size=parse_crop_size(args.crop_size),
            max_files=int(args.max_files),
            ref_source=str(args.ref_source),
            spike_reg=float(args.spike_reg),
            split_progress=bool(args.split_progress),
        )
    )
    return build_dataloader(dataset, args, shuffle=False)


def run_inference(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    history_length: int,
    saver: OutputSaver,
    method_name: str,
    sequence_mode: bool,
    save_every_n_frames: int,
    save_inputs: bool,
    save_gt: bool,
    amp: bool,
) -> None:
    model.eval()
    global_sample_idx = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc="inference")):
            left_seq = batch["left"].to(device, non_blocking=True)
            right_seq = batch["right"].to(device, non_blocking=True)
            f_clip = batch["f_clip"].to(device, non_blocking=True)
            mt = batch["mt"].to(device, non_blocking=True)
            target_a = batch["target_a"].to(device, non_blocking=True)
            target_b = batch["target_b"].to(device, non_blocking=True)
            threshold = batch["threshold"].to(device, non_blocking=True)
            mt = mt * threshold

            state_hist_left: list[StateEntry] = []
            state_hist_right: list[StateEntry] = []
            left_hist_imgs: list[torch.Tensor] = []
            right_hist_imgs: list[torch.Tensor] = []

            sequence_len = int(left_seq.shape[1])
            batch_size = int(left_seq.shape[0])

            for frame_idx in range(sequence_len):
                inputs = prepare_inputs(
                    left_seq[:, frame_idx],
                    right_seq[:, frame_idx],
                    f_clip,
                    state_hist_left,
                    state_hist_right,
                    sequence_mode,
                )
                inputs["target_a"] = None
                inputs["target_b"] = None
                inputs["ref"] = mt[:, frame_idx]

                if sequence_mode:
                    if left_hist_imgs:
                        inputs["left_history_imgs"] = tuple(left_hist_imgs)
                    if right_hist_imgs:
                        inputs["right_history_imgs"] = tuple(right_hist_imgs)

                amp_context = torch.cuda.amp.autocast() if amp and device.type == "cuda" else nullcontext()
                with amp_context:
                    outputs = model(inputs)

                pred_a = outputs["pred_a"].clamp(0.0, 1.0)
                pred_b = outputs["pred_b"].clamp(0.0, 1.0)

                update_history(state_hist_left, outputs.get("state_left"), history_length)
                update_history(state_hist_right, outputs.get("state_right"), history_length)
                append_and_truncate(left_hist_imgs, pred_a, history_length)
                append_and_truncate(right_hist_imgs, pred_b, history_length)

                if save_every_n_frames > 0 and frame_idx % save_every_n_frames == 0:
                    for b in range(batch_size):
                        save_frame_outputs(
                            saver=saver,
                            method_name=method_name,
                            sample_idx=global_sample_idx + b,
                            frame_idx=frame_idx,
                            pred_a=pred_a[b : b + 1],
                            pred_b=pred_b[b : b + 1],
                            input_a=left_seq[b : b + 1, frame_idx] if save_inputs else None,
                            input_b=right_seq[b : b + 1, frame_idx] if save_inputs else None,
                            gt_a=target_a[b : b + 1, frame_idx] if save_gt else None,
                            gt_b=target_b[b : b + 1, frame_idx] if save_gt else None,
                            ref=mt[b : b + 1, frame_idx],
                            pred_only=not (save_inputs or save_gt),
                            save_inputs_gt=save_inputs or save_gt,
                        )

            global_sample_idx += batch_size
            tqdm.write(f"[inference] batch={batch_idx:04d} sequence_len={sequence_len}")


def main() -> None:
    args = parse_inference_args()
    device = make_device(args)
    loader = _build_loader(args)

    model, sequence_mode = build_model(args)
    model = model.to(device)
    model = maybe_compile_model(model, args)

    checkpoint_path = resolve_checkpoint(args)
    if checkpoint_path is not None:
        print(f"[inference.py] Loading checkpoint: {checkpoint_path}")
        load_model_checkpoint(model, checkpoint_path, device)
    else:
        print("[inference.py] No checkpoint found/resolved. Running with current model weights.")

    os.makedirs(args.output_dir, exist_ok=True)
    saver = OutputSaver(SaveConfig(root_dir=args.output_dir, export_video=bool(args.export_videos)))

    print("=" * 80)
    print("[inference.py] Running spike inference")
    print(f"[inference.py] device: {device}")
    print(f"[inference.py] npz_path: {args.npz_path}")
    print(f"[inference.py] output_dir: {args.output_dir}")
    print(f"[inference.py] save_method_name: {args.save_method_name}")
    print(f"[inference.py] checkpoint: {checkpoint_path}")
    print(f"[inference.py] sequence_length: {args.sequence_length}")
    print(f"[inference.py] crop_size: {args.crop_size}")
    print(f"[inference.py] ref_source: {args.ref_source}")
    print(f"[inference.py] save_every_n_frames: {args.save_every_n_frames}")
    print("=" * 80)

    run_inference(
        model,
        loader,
        device=device,
        history_length=int(args.state_history),
        saver=saver,
        method_name=str(args.save_method_name),
        sequence_mode=sequence_mode,
        save_every_n_frames=int(args.save_every_n_frames),
        save_inputs=bool(args.save_inputs),
        save_gt=bool(args.save_gt),
        amp=bool(getattr(args, "amp", False)),
    )
    saver.close()

    print()
    print("=" * 80)
    print("[inference.py] Done")
    print(f"[inference.py] Saved outputs to: {args.output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
