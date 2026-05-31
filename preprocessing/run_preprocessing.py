"""Unified runner for MonoSpikeStereo preprocessing."""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import inspect
import json
import sys
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None


PREPROCESSING_DIR = Path(__file__).resolve().parent
CONFIG_DIR = PREPROCESSING_DIR / "config"
RAFT_DIR = PREPROCESSING_DIR / "RAFT"
RAFT_CORE_DIR = RAFT_DIR / "core"
PREPROCESSING_UTILS_DIR = PREPROCESSING_DIR / "utils"
DEFAULT_RAFT_MODEL = RAFT_DIR / "models" / "raft-things.pth"


@dataclass(frozen=True)
class PipelineSpec:
    name: str
    module_name: str
    mode: str
    description: str
    default_config_name: str
    builtin_defaults: dict[str, Any]


PIPELINES: dict[str, PipelineSpec] = {
    "tartanair_interpolate": PipelineSpec(
        name="tartanair_interpolate",
        module_name="interpolate_with_flow_tartanair",
        mode="interpolate",
        description="Generate interpolated TartanAir frame sequences with RAFT.",
        default_config_name="tartanair_interpolate.yaml",
        builtin_defaults={
            "input_root": "data/tartanair/raw",
            "output_root": "data/tartanair/interp",
            "raft_model_path": str(DEFAULT_RAFT_MODEL),
            "num_interpolations": 59,
            "crop_size": "320x320",
            "max_frames": None,
            "group_size": 3,
            "max_groups": 100,
            "group_skip": 4,
            "use_bidirectional_flow": True,
        },
    ),
    "tartanair_simulation": PipelineSpec(
        name="tartanair_simulation",
        module_name="pipelines.tartanair_simulation",
        mode="simulation",
        description="Generate mixed-spike NPZ files from interpolated TartanAir data.",
        default_config_name="tartanair_simulation.yaml",
        builtin_defaults={
            "root": "data/tartanair/interp",
            "out": "data/tartanair/sim",
            "threshold": 0.5,
            "length": 20,
            "noise": False,
            "noise_level": 0.01,
            "crop": "256x256",
            "random_crop": False,
            "lcd": True,
            "save_spike_sequences": True,
        },
    ),
}

PIPELINE_ALIASES = {
    "tartanair_flow": "tartanair_interpolate",
    "tartanair_spike": "tartanair_simulation",
}


def _prepend_sys_path(path: Path) -> None:
    path_str = str(path.resolve())
    if path.exists() and path_str not in sys.path:
        sys.path.insert(0, path_str)


def _bootstrap_for_simulation() -> None:
    _prepend_sys_path(PREPROCESSING_DIR)


def _bootstrap_for_interpolation() -> None:
    _prepend_sys_path(PREPROCESSING_DIR)
    _prepend_sys_path(PREPROCESSING_UTILS_DIR)
    _prepend_sys_path(RAFT_CORE_DIR)


def _resolve_pipeline_name(name: str) -> str:
    normalized = PIPELINE_ALIASES.get(name.strip().lower(), name.strip().lower())
    if normalized not in PIPELINES:
        supported = ", ".join(sorted(PIPELINES))
        raise ValueError(f"Unknown pipeline '{name}'. Supported values: {supported}")
    return normalized


def _default_config_path(spec: PipelineSpec) -> Path:
    return CONFIG_DIR / spec.default_config_name


def _load_yaml_or_json(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    if suffix in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError("PyYAML is needed to load YAML configs.")
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        return {} if data is None else dict(data)
    raise ValueError(f"Unsupported config format: {path}")


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = _deep_merge(out[key], value)  # type: ignore[arg-type]
        else:
            out[key] = value
    return out


def _coerce_value(text: str) -> Any:
    if yaml is not None:
        try:
            return yaml.safe_load(text)
        except Exception:
            return text
    lowered = text.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"null", "none"}:
        return None
    try:
        return float(text) if "." in text else int(text)
    except Exception:
        return text


def _set_nested_value(payload: MutableMapping[str, Any], dotted_key: str, value: Any) -> None:
    parts = [part for part in dotted_key.split(".") if part]
    if not parts:
        raise ValueError(f"Invalid override key: {dotted_key!r}")

    cursor: MutableMapping[str, Any] = payload
    for part in parts[:-1]:
        current = cursor.get(part)
        if not isinstance(current, MutableMapping):
            current = {}
            cursor[part] = current
        cursor = current
    cursor[parts[-1]] = value


def _apply_overrides(payload: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    result = dict(payload)
    for item in overrides:
        if "=" not in item:
            raise ValueError(
                f"Invalid override '{item}'. Expected KEY=VALUE, e.g. --set group_size=4"
            )
        key, raw_value = item.split("=", 1)
        _set_nested_value(result, key.strip(), _coerce_value(raw_value))
    return result


def _flatten_config_payload(data: Mapping[str, Any]) -> dict[str, Any]:
    if "params" in data and isinstance(data["params"], Mapping):
        merged = {k: v for k, v in data.items() if k != "params"}
        return _deep_merge(merged, data["params"])  # type: ignore[arg-type]
    return dict(data)


def _parse_hw(value: Any) -> tuple[int, int] | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return int(value[0]), int(value[1])
    if isinstance(value, str):
        raw = value.strip().lower().replace(",", "x")
        parts = [part.strip() for part in raw.split("x") if part.strip()]
        if len(parts) != 2:
            raise ValueError(f"Invalid size format: {value!r}. Expected HxW.")
        return int(parts[0]), int(parts[1])
    raise ValueError(f"Unsupported size value: {value!r}")


def _normalize_interpolation_config(cfg: dict[str, Any]) -> dict[str, Any]:
    out = dict(cfg)
    if "raft_model" in out and "raft_model_path" not in out:
        out["raft_model_path"] = out.pop("raft_model")
    if "single_flow" in out and "use_bidirectional_flow" not in out:
        out["use_bidirectional_flow"] = not bool(out.pop("single_flow"))
    if "crop_size" in out:
        out["crop_size"] = _parse_hw(out["crop_size"])
    return out


def _serialize_for_display(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.asdict(value)
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, Mapping):
        return {k: _serialize_for_display(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_serialize_for_display(v) for v in value]
    return value


def _load_pipeline_config(
    spec: PipelineSpec,
    config_path: str | None,
    overrides: list[str],
) -> dict[str, Any]:
    merged = dict(spec.builtin_defaults)
    candidate = Path(config_path).expanduser().resolve() if config_path else _default_config_path(spec)
    if candidate.exists():
        file_cfg = _flatten_config_payload(_load_yaml_or_json(candidate))
        merged = _deep_merge(merged, file_cfg)

    merged = _apply_overrides(merged, overrides)
    if spec.mode == "interpolate":
        merged = _normalize_interpolation_config(merged)

    merged["pipeline"] = spec.name
    merged["config_path"] = str(candidate) if candidate.exists() else None
    return merged


def _import_pipeline_module(spec: PipelineSpec):
    if spec.mode == "interpolate":
        _bootstrap_for_interpolation()
    elif spec.mode == "simulation":
        _bootstrap_for_simulation()
    else:
        raise ValueError(f"Unsupported pipeline mode: {spec.mode}")

    try:
        return importlib.import_module(spec.module_name)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to import preprocessing module '{spec.module_name}'."
        ) from exc


def _validate_interpolation_config(cfg: Mapping[str, Any]) -> None:
    input_root = cfg.get("input_root")
    if not input_root or not Path(str(input_root)).exists():
        raise FileNotFoundError(
            "Interpolation input_root does not exist. "
            "Set it in config or via --set input_root=/your/path"
        )

    raft_model = cfg.get("raft_model_path")
    if not raft_model or not Path(str(raft_model)).exists():
        raise FileNotFoundError(
            "RAFT checkpoint not found. "
            "Set raft_model_path in config or via --set raft_model_path=/path/to/raft-things.pth"
        )


def _validate_simulation_config(cfg: Mapping[str, Any]) -> None:
    root = cfg.get("root")
    if not root or not Path(str(root)).exists():
        raise FileNotFoundError(
            "Simulation root does not exist. Set it via config or --set root=/your/path"
        )


def _run_interpolation_pipeline(spec: PipelineSpec, cfg: dict[str, Any]) -> None:
    _validate_interpolation_config(cfg)
    module = _import_pipeline_module(spec)
    if not hasattr(module, "main"):
        raise AttributeError(f"Module '{spec.module_name}' does not expose main(...)")

    main_fn = module.main
    signature = inspect.signature(main_fn)
    kwargs = {key: value for key, value in cfg.items() if key in signature.parameters}
    missing = [
        name
        for name, parameter in signature.parameters.items()
        if parameter.default is inspect._empty and name not in kwargs
    ]
    if missing:
        raise ValueError(f"Missing parameters for {spec.name}: {', '.join(missing)}")
    main_fn(**kwargs)


def _run_simulation_pipeline(spec: PipelineSpec, cfg: dict[str, Any]) -> None:
    _validate_simulation_config(cfg)
    module = _import_pipeline_module(spec)
    if not hasattr(module, "Config") or not hasattr(module, "entry"):
        raise AttributeError(f"Module '{spec.module_name}' must expose Config and entry(cfg)")

    config_cls = module.Config
    entry_fn = module.entry
    field_names = {field.name for field in dataclasses.fields(config_cls)}
    config_obj = config_cls(**{key: value for key, value in cfg.items() if key in field_names})
    entry_fn(config_obj)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MonoSpikeStereo preprocessing runner.")
    parser.add_argument("--pipeline", default=None, help="Pipeline name.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional YAML / JSON config path. Defaults to preprocessing/config/<pipeline>.yaml.",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Override config values, e.g. --set group_size=4 --set crop_size=320x320",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print config without running.")
    parser.add_argument("--list-pipelines", action="store_true", help="List supported pipelines.")
    return parser


def _print_pipeline_list() -> None:
    print("Supported preprocessing pipelines:")
    for name in sorted(PIPELINES):
        spec = PIPELINES[name]
        print(f"- {name}")
        print(f"  mode: {spec.mode}")
        print(f"  description: {spec.description}")
        print(f"  default config: {_default_config_path(spec)}")


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    if args.list_pipelines:
        _print_pipeline_list()
        return 0
    if not args.pipeline:
        parser.error("--pipeline is needed unless --list-pipelines is used.")

    pipeline_name = _resolve_pipeline_name(args.pipeline)
    spec = PIPELINES[pipeline_name]
    cfg = _load_pipeline_config(spec, args.config, args.overrides)

    if args.dry_run:
        print(json.dumps(_serialize_for_display(cfg), indent=2, ensure_ascii=False))
        return 0

    print(f"[preprocessing] pipeline={spec.name}")
    if cfg.get("config_path"):
        print(f"[preprocessing] config={cfg['config_path']}")

    if spec.mode == "interpolate":
        _run_interpolation_pipeline(spec, cfg)
    elif spec.mode == "simulation":
        _run_simulation_pipeline(spec, cfg)
    else:
        raise ValueError(f"Unsupported pipeline mode: {spec.mode}")

    print("[preprocessing] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
