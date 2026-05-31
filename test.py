"""Test runner for trained MonoSpikeStereo checkpoints.

This script mirrors `train.py` setup and `src.train.engine.validate()` style
sequence inference, while additionally saving per-frame prediction images for
visual inspection.

Default behavior:
- builds dataloaders/model exactly like training
- loads checkpoint if provided / resolvable
- runs on the validation loader
- computes reconstruction loss + PSNR/SSIM
- saves input / prediction / GT images for every frame

Optional flags let you reduce saved content if needed.
"""

from __future__ import annotations

import argparse
import os
import tempfile
import warnings
from collections.abc import Iterable
from contextlib import nullcontext

os.environ.setdefault(
    "MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib-monospikestereo")
)
warnings.filterwarnings("ignore", category=FutureWarning, module=r"torchmetrics\..*")

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics.functional.image import (
    peak_signal_noise_ratio,
    structural_similarity_index_measure,
)
from tqdm import tqdm

from dataset.dataset import StereoSimulationDataset
from src.train.losses import build_reconstruction_loss
from src.train.setup import (
    build_model,
    make_device,
    maybe_compile_model,
)
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
    metric_scalar,
    prepare_inputs,
    update_history,
)
from utils.utils_train import parse_crop_size


def _build_test_extra_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        add_help=False,
        description="Validation-style test runner that saves input/prediction/GT images.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Explicit checkpoint path to load. Overrides --resume/--resume_path.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Root directory for saved test images.",
    )
    parser.add_argument(
        "--save_method_name",
        type=str,
        default="test",
        help="Subdirectory name used by OutputSaver.",
    )
    parser.add_argument(
        "--max_test_batches",
        type=int,
        default=0,
        help="Limit number of validation/test batches (<=0 uses max_val_batches / full val set).",
    )
    parser.add_argument(
        "--save_every_n_frames",
        type=int,
        default=1,
        help="Save predictions every N frames (1 means save all frames).",
    )
    add_bool_arg(
        parser,
        "pred_only",
        default=False,
        help_text="Only save prediction images.",
    )
    add_bool_arg(
        parser,
        "save_inputs_gt",
        default=True,
        help_text="Save input / GT images in addition to predictions.",
    )
    return parser


def parse_test_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    """Parse train args plus test-only extras."""
    extra_parser = _build_test_extra_parser()
    args = parse_train_args_with_extra_options(
        argv,
        extra_parser=extra_parser,
        options_title="Test options",
        default_output_root="test_outputs",
    )

    if int(args.max_test_batches) <= 0:
        args.max_test_batches = int(getattr(args, "max_val_batches", 0) or 0)

    return args


def _build_test_dataloader(args: argparse.Namespace) -> DataLoader:
    """Build a deterministic external-test loader from all NPZ files in npz_path."""
    crop = parse_crop_size(args.crop_size)
    dataset = StereoSimulationDataset(
        args.npz_path,
        sequence_length=args.sequence_length,
        split="test_all",
        crop_size=crop,
    )
    return build_dataloader(dataset, args, shuffle=False)


def test_and_save(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    recon_criterion: nn.Module,
    device: torch.device,
    history_length: int,
    saver: OutputSaver,
    *,
    method_name: str = "test",
    sequence_mode: bool = True,
    max_test_batches: int = 0,
    save_every_n_frames: int = 1,
    pred_only: bool = True,
    save_inputs_gt: bool = False,
    amp: bool = False,
) -> tuple[float, float, float, float, float]:
    """Validation-style loop with output image saving."""
    orig_training = model.training
    model.eval()

    running_loss = 0.0
    running_psnr_a = 0.0
    running_psnr_b = 0.0
    running_ssim_a = 0.0
    running_ssim_b = 0.0
    frame_counter = 0

    global_sample_idx = 0
    total_batches = None
    if max_test_batches > 0:
        total_batches = min(len(loader), max_test_batches)

    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc="test", total=total_batches)):
            left_seq = batch["left"].to(device, non_blocking=True)
            right_seq = batch["right"].to(device, non_blocking=True)
            f_clip = batch["f_clip"].to(device, non_blocking=True)
            target_a = batch["target_a"].to(device, non_blocking=True)
            target_b = batch["target_b"].to(device, non_blocking=True)
            mt = batch["mt"].to(device, non_blocking=True)
            threshold = batch["threshold"].to(device, non_blocking=True)
            mt = mt * threshold

            state_hist_left: list[StateEntry] = []
            state_hist_right: list[StateEntry] = []
            left_hist_imgs: list[torch.Tensor] = []
            right_hist_imgs: list[torch.Tensor] = []

            sequence_len = int(left_seq.shape[1])
            batch_size = int(left_seq.shape[0])

            for j in range(sequence_len):
                inputs = prepare_inputs(
                    left_seq[:, j],
                    right_seq[:, j],
                    f_clip,
                    state_hist_left,
                    state_hist_right,
                    sequence_mode,
                )
                inputs["target_a"] = None
                inputs["target_b"] = None
                inputs["ref"] = mt[:, j]

                if sequence_mode:
                    if len(left_hist_imgs) > 0:
                        inputs["left_history_imgs"] = tuple(left_hist_imgs)
                    if len(right_hist_imgs) > 0:
                        inputs["right_history_imgs"] = tuple(right_hist_imgs)

                amp_context = (
                    torch.cuda.amp.autocast()
                    if amp and device.type == "cuda"
                    else nullcontext()
                )
                with amp_context:
                    outputs = model(inputs)

                pred_a = outputs["pred_a"].clamp(0.0, 1.0)
                pred_b = outputs["pred_b"].clamp(0.0, 1.0)

                update_history(state_hist_left, outputs.get("state_left"), history_length)
                update_history(state_hist_right, outputs.get("state_right"), history_length)
                append_and_truncate(left_hist_imgs, pred_a, history_length)
                append_and_truncate(right_hist_imgs, pred_b, history_length)

                recon_loss = recon_criterion(pred_a, target_a[:, j]) + recon_criterion(
                    pred_b, target_b[:, j]
                )
                running_loss += float(recon_loss.item())
                frame_counter += 1

                psnr_a_raw = peak_signal_noise_ratio(
                    pred_a,
                    target_a[:, j],
                    data_range=1.0,
                )
                psnr_b_raw = peak_signal_noise_ratio(
                    pred_b,
                    target_b[:, j],
                    data_range=1.0,
                )
                running_psnr_a += float(
                    metric_scalar(psnr_a_raw, device=device, dtype=pred_a.dtype).item()
                )
                running_psnr_b += float(
                    metric_scalar(psnr_b_raw, device=device, dtype=pred_b.dtype).item()
                )

                pred_a_4d = pred_a if pred_a.dim() == 4 else pred_a.unsqueeze(1)
                target_a_4d = (
                    target_a[:, j]
                    if target_a[:, j].dim() == 4
                    else target_a[:, j].unsqueeze(1)
                )
                pred_b_4d = pred_b if pred_b.dim() == 4 else pred_b.unsqueeze(1)
                target_b_4d = (
                    target_b[:, j]
                    if target_b[:, j].dim() == 4
                    else target_b[:, j].unsqueeze(1)
                )

                ssim_a_raw = structural_similarity_index_measure(
                    pred_a_4d,
                    target_a_4d,
                    data_range=1.0,
                )
                ssim_b_raw = structural_similarity_index_measure(
                    pred_b_4d,
                    target_b_4d,
                    data_range=1.0,
                )
                running_ssim_a += float(
                    metric_scalar(ssim_a_raw, device=device, dtype=pred_a.dtype).item()
                )
                running_ssim_b += float(
                    metric_scalar(ssim_b_raw, device=device, dtype=pred_b.dtype).item()
                )

                should_save = save_every_n_frames > 0 and (j % save_every_n_frames == 0)
                if should_save:
                    for b in range(batch_size):
                        sample_idx = global_sample_idx + b
                        save_frame_outputs(
                            saver=saver,
                            method_name=method_name,
                            sample_idx=sample_idx,
                            frame_idx=j,
                            pred_a=pred_a[b : b + 1],
                            pred_b=pred_b[b : b + 1],
                            input_a=left_seq[b : b + 1, j] if (not pred_only or save_inputs_gt) else None,
                            input_b=right_seq[b : b + 1, j] if (not pred_only or save_inputs_gt) else None,
                            gt_a=target_a[b : b + 1, j] if (not pred_only or save_inputs_gt) else None,
                            gt_b=target_b[b : b + 1, j] if (not pred_only or save_inputs_gt) else None,
                            pred_only=pred_only,
                            save_inputs_gt=save_inputs_gt,
                        )

            global_sample_idx += batch_size

            avg_loss_so_far = running_loss / max(1, frame_counter)
            avg_psnr_a_so_far = running_psnr_a / max(1, frame_counter)
            avg_psnr_b_so_far = running_psnr_b / max(1, frame_counter)
            tqdm.write(
                f"[test] batch={i:04d} "
                f"loss={avg_loss_so_far:.4f} "
                f"psnr_a={avg_psnr_a_so_far:.2f} "
                f"psnr_b={avg_psnr_b_so_far:.2f}"
            )

            if max_test_batches > 0 and (i + 1) >= max_test_batches:
                break

    avg_loss = running_loss / max(1, frame_counter)
    avg_psnr_a = running_psnr_a / max(1, frame_counter)
    avg_psnr_b = running_psnr_b / max(1, frame_counter)
    avg_ssim_a = running_ssim_a / max(1, frame_counter)
    avg_ssim_b = running_ssim_b / max(1, frame_counter)

    model.train(orig_training)

    return (
        avg_loss,
        avg_psnr_a,
        avg_psnr_b,
        avg_ssim_a,
        avg_ssim_b,
    )


def main() -> None:
    args = parse_test_args()
    device = make_device(args)

    test_loader = _build_test_dataloader(args)

    model, sequence_mode = build_model(args)
    model = model.to(device)
    model = maybe_compile_model(model, args)

    recon_criterion = build_reconstruction_loss(
        device=device,
        l1_weight=1.0,
        ssim_weight=0.1,
        perceptual_weight=0.0,
    )

    checkpoint_path = resolve_checkpoint(args)
    if checkpoint_path is not None:
        print(f"[test.py] Loading checkpoint: {checkpoint_path}")
        load_model_checkpoint(model, checkpoint_path, device)
    else:
        print("[test.py] No checkpoint found/resolved. Running with current model weights.")

    os.makedirs(args.output_dir, exist_ok=True)
    saver = OutputSaver(SaveConfig(root_dir=args.output_dir))

    print("=" * 80)
    print("[test.py] Running validation-style test")
    print(f"[test.py] device: {device}")
    print(f"[test.py] npz_path: {args.npz_path}")
    print(f"[test.py] output_dir: {args.output_dir}")
    print(f"[test.py] save_method_name: {args.save_method_name}")
    print(f"[test.py] checkpoint: {checkpoint_path}")
    print(f"[test.py] batch_size: {args.batch_size}")
    print(f"[test.py] sequence_length: {args.sequence_length}")
    print(f"[test.py] state_history: {args.state_history}")
    print(f"[test.py] pred_only: {args.pred_only}")
    print(f"[test.py] save_inputs_gt: {args.save_inputs_gt}")
    print(f"[test.py] save_every_n_frames: {args.save_every_n_frames}")
    print(f"[test.py] max_test_batches: {args.max_test_batches}")
    print("=" * 80)

    metrics = test_and_save(
        model=model,
        loader=test_loader,
        recon_criterion=recon_criterion,
        device=device,
        history_length=args.state_history,
        saver=saver,
        method_name=args.save_method_name,
        sequence_mode=sequence_mode,
        max_test_batches=int(args.max_test_batches),
        save_every_n_frames=int(args.save_every_n_frames),
        pred_only=bool(args.pred_only),
        save_inputs_gt=bool(args.save_inputs_gt),
        amp=bool(getattr(args, "amp", False)),
    )

    val_loss, val_psnr_a, val_psnr_b, val_ssim_a, val_ssim_b = metrics

    print()
    print("=" * 80)
    print("[test.py] Done")
    print(
        f"val_loss={val_loss:.4f}, "
        f"val_psnr_a={val_psnr_a:.2f}, val_psnr_b={val_psnr_b:.2f}, "
        f"val_ssim_a={val_ssim_a:.4f}, val_ssim_b={val_ssim_b:.4f}"
    )
    print(f"[test.py] Saved outputs to: {args.output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
