"""Small argparse helpers shared by train/test/inference entrypoints."""

from __future__ import annotations

import argparse


def add_bool_arg(
    parser: argparse.ArgumentParser,
    name: str,
    default: bool,
    help_text: str,
) -> None:
    """Add a boolean flag with --x and --no-x forms."""
    flag = f"--{name.replace('_', '-')}"
    no_flag = f"--no-{name.replace('_', '-')}"

    if hasattr(argparse, "BooleanOptionalAction"):
        parser.add_argument(
            flag,
            dest=name,
            action=argparse.BooleanOptionalAction,
            default=default,
            help=help_text,
        )
        return

    parser.add_argument(flag, dest=name, action="store_true", help=help_text)
    parser.add_argument(no_flag, dest=name, action="store_false", help=help_text)
    parser.set_defaults(**{name: default})
