"""Shared helpers for script entrypoints built on the training config parser."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Iterable

from src.train.config import parse_args as parse_train_args
from src.train.setup import resolve_resume_checkpoint


def parse_train_args_with_extra_options(
    argv: Iterable[str] | None,
    *,
    extra_parser: argparse.ArgumentParser,
    options_title: str,
    default_output_root: str,
) -> argparse.Namespace:
    """Parse train args plus script-specific extra options."""
    arg_list = list(argv) if argv is not None else sys.argv[1:]

    if any(arg in {"-h", "--help"} for arg in arg_list):
        try:
            parse_train_args(["--help"])
        except SystemExit:
            pass
        print(f"\n{options_title}:")
        extra_parser.print_help()
        raise SystemExit(0)

    args = parse_train_args(arg_list)

    extra_defaults = vars(extra_parser.parse_args([]))
    for key, value in extra_defaults.items():
        if not hasattr(args, key):
            setattr(args, key, value)

    # Validate script-specific CLI options without replacing YAML values with
    # parser defaults. Generic YAML overrides are already handled above.
    extra_parser.parse_known_args(arg_list)

    if args.output_dir is None:
        args.output_dir = os.path.join(default_output_root, f"{args.net_type}_{args.name}")

    return args


def resolve_checkpoint(args: argparse.Namespace) -> str | None:
    """Resolve an explicit, resumed, or best checkpoint path."""
    if getattr(args, "checkpoint", None):
        return str(args.checkpoint)

    resume_ckpt = resolve_resume_checkpoint(args)
    if resume_ckpt is not None:
        return str(resume_ckpt)

    best_ckpt = os.path.join(args.checkpoint_dir, f"best_{args.net_type}_{args.name}.pth")
    if os.path.exists(best_ckpt):
        return best_ckpt

    return None
