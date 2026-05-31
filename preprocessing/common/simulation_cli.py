"""Shared CLI helpers for preprocessing simulation pipelines."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any, Mapping, TypeVar

from runtime import add_config_argument, apply_parser_defaults, load_cli_config

ConfigT = TypeVar("ConfigT", bound="SimulationConfig")


@dataclass
class SimulationConfig:
    """Common configuration for spike-simulation preprocessing pipelines."""

    root: str = "./"
    out: str = "./spike_syn_data/output"
    threshold: float = 0.5
    length: int = 30
    noise: bool = False
    noise_level: float = 0.1
    crop: str | None = None  # legacy format: WxH
    random_crop: bool = False
    lcd: bool = False
    save_spike_sequences: bool = True


SIMULATION_META_KEYS = {"pipeline", "dataset", "description", "config_path", "params"}


def flatten_simulation_config_defaults(raw_cfg: Mapping[str, Any] | None) -> dict[str, Any]:
    """Flatten a loaded preprocessing config."""
    if raw_cfg is None:
        return {}

    if not isinstance(raw_cfg, Mapping):
        return {}

    params = raw_cfg.get("params")
    if isinstance(params, Mapping):
        merged = dict(params)
        for key, value in raw_cfg.items():
            if key not in SIMULATION_META_KEYS and key not in merged:
                merged[key] = value
        return merged

    return {
        key: value
        for key, value in raw_cfg.items()
        if key not in SIMULATION_META_KEYS
    }


def parse_legacy_crop(value: str | None) -> tuple[int, int] | None:
    """Parse ``WxH`` into ``(H, W)``."""
    if value is None:
        return None

    text = value.strip().lower()
    if not text:
        return None

    if "x" in text:
        parts = text.split("x")
    elif "," in text:
        parts = text.split(",")
    else:
        raise ValueError("crop format must be WxH or W,H (e.g. 256x256)")

    if len(parts) != 2:
        raise ValueError("crop format must contain exactly two integers")

    width, height = int(parts[0]), int(parts[1])
    if width <= 0 or height <= 0:
        raise ValueError("crop dimensions must be positive")

    return height, width


def build_simulation_parser(
    *,
    default_config: str,
    default_root: str,
    default_out: str,
    default_crop: str | None,
    description: str = "Spike-style preprocessing simulation pipeline.",
) -> argparse.ArgumentParser:
    """Build a shared argparse parser for simulation pipelines."""
    parser = argparse.ArgumentParser(description=description)
    add_config_argument(parser, default=default_config)

    parser.add_argument("--root", type=str, default=default_root)
    parser.add_argument("--out", type=str, default=default_out)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--length", type=int, default=20)

    parser.add_argument("--noise", dest="noise", action="store_true")
    parser.add_argument("--no-noise", dest="noise", action="store_false")
    parser.set_defaults(noise=False)

    parser.add_argument("--noise-level", type=float, default=0.01)
    parser.add_argument("--crop", type=str, default=default_crop)

    parser.add_argument("--random-crop", dest="random_crop", action="store_true")
    parser.add_argument("--no-random-crop", dest="random_crop", action="store_false")
    parser.set_defaults(random_crop=False)

    parser.add_argument("--lcd", dest="lcd", action="store_true")
    parser.add_argument("--no-lcd", dest="lcd", action="store_false")
    parser.set_defaults(lcd=True)

    parser.add_argument(
        "--save-spike-sequences",
        dest="save_spike_sequences",
        action="store_true",
    )
    parser.add_argument(
        "--no-save-spike-sequences",
        dest="save_spike_sequences",
        action="store_false",
    )
    parser.set_defaults(save_spike_sequences=True)

    return parser


def parse_simulation_args(
    argv: list[str] | None = None,
    *,
    default_config: str,
    default_root: str,
    default_out: str,
    default_crop: str | None,
    description: str = "Spike-style preprocessing simulation pipeline.",
) -> argparse.Namespace:
    """Parse shared simulation CLI arguments with YAML-config defaults."""
    raw_cfg, pre_args = load_cli_config(argv, default_config=default_config)
    cfg_defaults = flatten_simulation_config_defaults(raw_cfg)

    parser = build_simulation_parser(
        default_config=default_config,
        default_root=default_root,
        default_out=default_out,
        default_crop=default_crop,
        description=description,
    )

    parser.set_defaults(config=pre_args.config)
    apply_parser_defaults(parser, cfg_defaults)

    return parser.parse_args(argv)


def namespace_to_simulation_config(
    args: argparse.Namespace,
    *,
    config_cls: type[ConfigT] = SimulationConfig,
) -> ConfigT:
    """Convert parsed argparse args into a simulation config dataclass."""
    return config_cls(
        root=str(args.root),
        out=str(args.out),
        threshold=float(args.threshold),
        length=int(args.length),
        noise=bool(args.noise),
        noise_level=float(args.noise_level),
        crop=None if args.crop is None else str(args.crop),
        random_crop=bool(args.random_crop),
        lcd=bool(args.lcd),
        save_spike_sequences=bool(args.save_spike_sequences),
    )


__all__ = [
    "SIMULATION_META_KEYS",
    "SimulationConfig",
    "build_simulation_parser",
    "flatten_simulation_config_defaults",
    "namespace_to_simulation_config",
    "parse_legacy_crop",
    "parse_simulation_args",
]
