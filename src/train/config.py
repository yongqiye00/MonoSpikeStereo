"""YAML-first training config loader.

Configuration should normally live in YAML files. The command line only selects
the config file and optionally applies light one-off overrides as ``--key value``
or ``--no-key``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

import torch

from src.common.config_io import load_defaults


DEFAULT_CONFIG_FILENAME = "defaults.yaml"
_MODULE_DIR = Path(__file__).resolve().parent

# Built-in fallback defaults. A config file can override any of these values.
BUILTIN_DEFAULTS: Dict[str, Any] = {
    "npz_path": "data/train_npz",
    "log_dir": "runs/exp_seq",
    "batch_size": 4,
    "lr": 1e-5,
    "grad_clip_norm": 1.0,
    "epochs": 2000,
    "save_every_epochs": 5,
    "max_train_samples": 0,
    "max_val_batches": 0,
    "sequence_length": 18,
    "train_clip_stride": 1,
    "state_history": 4,
    "num_workers": 8,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "checkpoint_dir": "checkpoints",
    "net_type": "flow",
    "resume": False,
    "resume_path": None,
    "init_checkpoint": None,
    "crop_size": "128x256",
    "compile": False,
    "amp": False,
    "amp_dtype": "fp16",
    "name": "unnamed_seq",
}


def _load_defaults(config_path: Optional[str]) -> Tuple[Dict[str, Any], Optional[Path]]:
    merged, resolved = load_defaults(
        builtin_defaults=BUILTIN_DEFAULTS,
        config_path=config_path,
        module_dir=_MODULE_DIR,
        default_filename=DEFAULT_CONFIG_FILENAME,
    )
    return merged, resolved


def _parse_scalar(raw: str) -> Any:
    value = raw.strip()
    lowered = value.lower()
    if lowered in {"none", "null"}:
        return None
    if lowered in {"true", "false"}:
        return lowered == "true"

    try:
        return int(value)
    except ValueError:
        pass

    try:
        return float(value)
    except ValueError:
        return raw


def _coerce_override_value(key: str, raw: str, cfg: Mapping[str, Any]) -> Any:
    current = cfg.get(key)
    if current is None:
        return _parse_scalar(raw)
    if isinstance(current, bool):
        parsed = _parse_scalar(raw)
        if isinstance(parsed, bool):
            return parsed
        lowered = raw.strip().lower()
        if lowered in {"1", "yes", "y", "on"}:
            return True
        if lowered in {"0", "no", "n", "off"}:
            return False
        raise ValueError(f"Cannot parse boolean override for {key!r}: {raw!r}")
    if isinstance(current, int) and not isinstance(current, bool):
        return int(raw)
    if isinstance(current, float):
        return float(raw)
    return raw


def _parse_cli_overrides(tokens: list[str], cfg: Mapping[str, Any]) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if not token.startswith("--"):
            raise ValueError(f"Unexpected positional argument: {token}")

        name_value = token[2:]
        if "=" in name_value:
            name, raw = name_value.split("=", 1)
            key = name.replace("-", "_")
            overrides[key] = _coerce_override_value(key, raw, cfg)
            i += 1
            continue

        if name_value.startswith("no-"):
            key = name_value[3:].replace("-", "_")
            overrides[key] = False
            i += 1
            continue

        key = name_value.replace("-", "_")
        if i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
            overrides[key] = _coerce_override_value(key, tokens[i + 1], cfg)
            i += 2
        else:
            overrides[key] = True
            i += 1

    return overrides


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    """Load YAML config and apply optional CLI overrides."""
    arg_list = list(argv) if argv is not None else None

    parser = argparse.ArgumentParser(
        description="MonoSpikeStereo YAML configuration.",
        epilog=(
            "Configuration values should normally be edited in YAML. "
            "For one-off runs, any config key may still be overridden as "
            "`--key value`, `--key=value`, or `--no-key`."
        ),
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file (.json/.yaml/.yml).",
    )
    parsed, unknown = parser.parse_known_args(arg_list)

    cfg, resolved_config_path = _load_defaults(parsed.config)
    cfg.update(_parse_cli_overrides(unknown, cfg))

    args = argparse.Namespace(**cfg)
    args.config_path = str(resolved_config_path) if resolved_config_path else None
    return args
