"""Shared config loading utilities for JSON/YAML defaults.

This module centralizes common logic used by train/eval config loaders:
- resolve optional config paths
- load JSON/YAML config files
- validate top-level mapping payloads
- deep-merge user config onto built-in defaults
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover
    yaml = None


def _as_dict(data: Any, *, source: Path) -> Dict[str, Any]:
    """Validate config payload is a mapping and return a plain dict."""
    if data is None:
        return {}
    if not isinstance(data, Mapping):
        raise ValueError(f"Config at {source} must be a mapping/object at top level.")
    return dict(data)


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    """Recursively merge `override` onto `base` and return a new dict."""
    out: Dict[str, Any] = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = deep_merge(dict(out[key]), value)  # type: ignore[index]
        else:
            out[key] = value
    return out


def read_json_config(path: Path) -> Dict[str, Any]:
    """Read and validate a JSON config file."""
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return _as_dict(payload, source=path)


def read_yaml_config(path: Path) -> Dict[str, Any]:
    """Read and validate a YAML config file."""
    if yaml is None:
        raise ImportError(
            "PyYAML is required to read YAML config files. Install `pyyaml` first."
        )
    with path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f)
    return _as_dict(payload, source=path)


def read_config_file(path: Path) -> Dict[str, Any]:
    """Read config by file suffix.

    Supported suffixes:
    - JSON: .json
    - YAML: .yaml / .yml
    """
    suffix = path.suffix.lower()
    if suffix == ".json":
        return read_json_config(path)
    if suffix in {".yaml", ".yml"}:
        return read_yaml_config(path)
    raise ValueError(
        f"Unsupported config format for {path}. Use .json, .yaml, or .yml."
    )


def resolve_config_path(
    config_path: Optional[str],
    *,
    module_dir: Path,
    default_filename: str,
) -> Optional[Path]:
    """Resolve user-provided config path or fallback default path.

    Returns:
        - resolved path if found
        - None if no path specified and default file does not exist
    """
    if config_path:
        resolved = Path(config_path).expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Config file not found: {resolved}")
        return resolved

    default_path = (module_dir / default_filename).resolve()
    return default_path if default_path.exists() else None


def load_defaults(
    *,
    builtin_defaults: Mapping[str, Any],
    config_path: Optional[str],
    module_dir: Path,
    default_filename: str,
) -> Tuple[Dict[str, Any], Optional[Path]]:
    """Load defaults by deep-merging file config onto built-in defaults.

    Precedence:
    1) file config (if present)
    2) built-in defaults
    """
    resolved = resolve_config_path(
        config_path,
        module_dir=module_dir,
        default_filename=default_filename,
    )
    if resolved is None:
        return dict(builtin_defaults), None

    file_cfg = read_config_file(resolved)
    merged = deep_merge(dict(builtin_defaults), file_cfg)
    return merged, resolved


def cfg_value(cfg: Mapping[str, Any], key: str, fallback: Any) -> Any:
    """Simple helper for retrieving top-level config values with fallback."""
    return cfg.get(key, fallback)
