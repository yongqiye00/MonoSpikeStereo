"""Runtime helpers for preprocessing scripts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Iterable

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


PREPROCESSING_ROOT = Path(__file__).resolve().parent
CONFIG_ROOT = PREPROCESSING_ROOT / "config"

RAFT_ROOT = PREPROCESSING_ROOT / "RAFT"
RAFT_CORE_DIR = RAFT_ROOT / "core"
RAFT_UTILS_DIR = RAFT_CORE_DIR / "utils"

LOCAL_UTILS_DIR = PREPROCESSING_ROOT / "utils"

DEFAULT_RAFT_CHECKPOINT_CANDIDATES = (
    RAFT_ROOT / "models" / "raft-things.pth",
    PREPROCESSING_ROOT / "models" / "raft-things.pth",
)


def _insert_sys_path(path: Path, *, prepend: bool = True) -> None:
    """Insert a path into ``sys.path`` only once."""
    resolved = str(path.resolve())
    if resolved in sys.path:
        return
    if prepend:
        sys.path.insert(0, resolved)
    else:
        sys.path.append(resolved)


def bootstrap_preprocessing_paths(
    *,
    add_local_utils: bool = True,
    add_raft: bool = False,
) -> None:
    """Bootstrap local import paths."""
    if add_local_utils:
        _insert_sys_path(PREPROCESSING_ROOT, prepend=True)

    if add_raft:
        if RAFT_UTILS_DIR.exists():
            _insert_sys_path(RAFT_UTILS_DIR, prepend=True)
        if RAFT_CORE_DIR.exists():
            _insert_sys_path(RAFT_CORE_DIR, prepend=True)


def bootstrap_simulation_runtime() -> None:
    """Bootstrap imports needed by simulation-style preprocessing scripts."""
    bootstrap_preprocessing_paths(add_local_utils=True, add_raft=False)


def bootstrap_interpolation_runtime() -> None:
    """Bootstrap imports needed by RAFT interpolation preprocessing scripts."""
    bootstrap_preprocessing_paths(add_local_utils=True, add_raft=True)


def ensure_package_root(path: Path, *, name: str) -> Path:
    """Validate a local dependency root exists."""
    if not path.exists():
        raise FileNotFoundError(f"Preprocessing dependency '{name}' was not found at: {path}")
    return path


def ensure_raft_available() -> Path:
    """Ensure the vendored RAFT source root exists."""
    return ensure_package_root(RAFT_ROOT, name="RAFT")


def default_raft_checkpoint() -> str | None:
    """Return the first existing default RAFT checkpoint path, if any."""
    for candidate in DEFAULT_RAFT_CHECKPOINT_CANDIDATES:
        if candidate.exists():
            return str(candidate)
    return None


def resolve_raft_checkpoint(cli_value: str | None = None) -> str:
    """Resolve the RAFT checkpoint path."""
    if cli_value:
        path = Path(cli_value).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"RAFT checkpoint not found: {path}")
        return str(path)

    default_path = default_raft_checkpoint()
    if default_path is not None:
        return default_path

    raise FileNotFoundError(
        "RAFT checkpoint was not provided and no default checkpoint was found. "
        f"Expected one of: {', '.join(str(p) for p in DEFAULT_RAFT_CHECKPOINT_CANDIDATES)}"
    )


def parse_hw_size(value: str | None) -> tuple[int, int] | None:
    """Parse a size string like ``256x256`` or ``256,256`` into ``(H, W)``."""
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
        raise ValueError(f"Invalid size format: {value!r}. Expected HxW or H,W.")

    if len(parts) != 2:
        raise ValueError(f"Invalid size format: {value!r}. Expected exactly two integers.")

    height, width = (int(parts[0]), int(parts[1]))
    if height <= 0 or width <= 0:
        raise ValueError(f"Invalid non-positive size: {value!r}")

    return height, width


def parse_wh_crop(value: str | None) -> tuple[int, int] | None:
    """Parse legacy ``WxH`` crop strings into ``(H, W)``."""
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
        raise ValueError(f"Invalid crop format: {value!r}. Expected WxH or W,H.")

    if len(parts) != 2:
        raise ValueError(f"Invalid crop format: {value!r}. Expected exactly two integers.")

    width, height = (int(parts[0]), int(parts[1]))
    if height <= 0 or width <= 0:
        raise ValueError(f"Invalid non-positive crop: {value!r}")

    return height, width


def resolve_config_path(config: str | None) -> Path | None:
    """Resolve a preprocessing config path."""
    if not config:
        return None

    raw = Path(config).expanduser()
    if raw.exists():
        return raw.resolve()

    candidate = CONFIG_ROOT / config
    if candidate.exists():
        return candidate.resolve()

    if candidate.suffix == "":
        for suffix in (".yaml", ".yml"):
            alt = candidate.with_suffix(suffix)
            if alt.exists():
                return alt.resolve()

    raise FileNotFoundError(
        f"Preprocessing config not found: {config}. "
        f"Searched direct path and under {CONFIG_ROOT}"
    )


def load_yaml_config(config: str | Path | None) -> dict[str, Any]:
    """Load a YAML config file as a plain dictionary."""
    if config is None:
        return {}

    if yaml is None:
        raise ImportError("PyYAML is needed to read preprocessing config files.")

    path = resolve_config_path(str(config)) if not isinstance(config, Path) else config
    if path is None:
        return {}

    with path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f)

    if payload is None:
        return {}

    if not isinstance(payload, dict):
        raise ValueError(f"Config at {path} must be a mapping/object at top level.")

    return dict(payload)


def load_cli_config(
    argv: Iterable[str] | None = None,
    *,
    default_config: str | None = None,
) -> tuple[dict[str, Any], argparse.Namespace]:
    """Pre-parse ``--config`` and load that YAML file."""
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument(
        "--config",
        type=str,
        default=default_config,
        help="Path or name of a preprocessing YAML config file.",
    )
    pre_args, _ = pre_parser.parse_known_args(list(argv) if argv is not None else None)
    cfg = load_yaml_config(pre_args.config)
    return cfg, pre_args


def apply_parser_defaults(
    parser: argparse.ArgumentParser,
    config: dict[str, Any],
    *,
    key_map: dict[str, str] | None = None,
) -> None:
    """Apply YAML config values to an ``argparse`` parser."""
    mapped: dict[str, Any] = {}
    key_map = key_map or {}

    for key, value in config.items():
        dest = key_map.get(key, key)
        mapped[dest] = value

    if mapped:
        parser.set_defaults(**mapped)


def add_config_argument(parser: argparse.ArgumentParser, *, default: str | None = None) -> None:
    """Add a standard ``--config`` option to a parser."""
    parser.add_argument(
        "--config",
        type=str,
        default=default,
        help="Path or name of a preprocessing YAML config file.",
    )


def namespace_to_clean_dict(namespace: argparse.Namespace) -> dict[str, Any]:
    """Convert an argparse namespace into a plain dictionary without private keys."""
    return {
        key: value
        for key, value in vars(namespace).items()
        if not key.startswith("_")
    }


def merge_config_overrides(
    base: dict[str, Any],
    overrides: dict[str, Any],
) -> dict[str, Any]:
    """Shallow-merge CLI/config dictionaries with ``overrides`` taking priority."""
    merged = dict(base)
    merged.update(overrides)
    return merged


__all__ = [
    "CONFIG_ROOT",
    "DEFAULT_RAFT_CHECKPOINT_CANDIDATES",
    "LOCAL_UTILS_DIR",
    "PREPROCESSING_ROOT",
    "RAFT_CORE_DIR",
    "RAFT_ROOT",
    "RAFT_UTILS_DIR",
    "add_config_argument",
    "apply_parser_defaults",
    "bootstrap_interpolation_runtime",
    "bootstrap_preprocessing_paths",
    "bootstrap_simulation_runtime",
    "default_raft_checkpoint",
    "load_cli_config",
    "load_yaml_config",
    "merge_config_overrides",
    "namespace_to_clean_dict",
    "parse_hw_size",
    "parse_wh_crop",
    "ensure_raft_available",
    "resolve_config_path",
    "resolve_raft_checkpoint",
]
