"""Layered configuration with capability/model precedence.

Effective settings are resolved with this decreasing precedence:

    CLI args > env vars (LOCALAI_*) > config file
        (per-model table > per-capability table > global [defaults])
    > built-in defaults (model-spec default > capability default > core default)

The result is a :class:`Settings` object with typed, validated accessors.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from localai.core.errors import InvalidArgumentError
from localai.core.registry import ModelSpec

# Universal (core) defaults — lowest precedence of all.
CORE_DEFAULTS: Dict[str, Any] = {
    "output_dir": "outputs",
    "offload": "none",
    "attention": "sdpa",
    "batch": 1,
    "seed": None,
    "dtype": "bfloat16",
}

# Known typed keys; used to coerce env/file strings and validate.
KEY_TYPES: Dict[str, type] = {
    "steps": int,
    "width": int,
    "height": int,
    "batch": int,
    "seed": int,
    "max_sequence_length": int,
    "guidance": float,
    "model": str,
    "preset": str,
    "dtype": str,
    "offload": str,
    "attention": str,
    "output_dir": str,
    "negative_prompt": str,
}

# Candidate config file names searched in the CWD when none is supplied.
_DEFAULT_CONFIG_NAMES = ("localai.toml", "config.toml")


def _coerce(key: str, value: Any) -> Any:
    """Coerce *value* to the declared type for *key*; raise on bad input."""
    if value is None:
        return None
    target = KEY_TYPES.get(key)
    if target is None or isinstance(value, target):
        return value
    try:
        if target is int:
            return int(str(value).strip())
        if target is float:
            return float(str(value).strip())
        if target is str:
            return str(value)
    except (TypeError, ValueError):
        raise InvalidArgumentError(
            f"invalid value for '{key}': {value!r} (expected {target.__name__})"
        )
    return value


@dataclass
class Settings:
    """Resolved effective settings for one ``(capability, model)`` run."""

    capability_id: str
    model_id: str
    values: Dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        val = self.values.get(key, default)
        return _coerce(key, val) if val is not None else default

    def get_int(self, key: str, default: Optional[int] = None) -> Optional[int]:
        val = self.get(key, default)
        return None if val is None else int(val)

    def get_float(self, key: str, default: Optional[float] = None) -> Optional[float]:
        val = self.get(key, default)
        return None if val is None else float(val)

    def get_str(self, key: str, default: Optional[str] = None) -> Optional[str]:
        val = self.get(key, default)
        return None if val is None else str(val)

    def get_bool(self, key: str, default: bool = False) -> bool:
        val = self.values.get(key, default)
        if isinstance(val, bool):
            return val
        if isinstance(val, str):
            return val.strip().lower() in ("1", "true", "yes", "on")
        return bool(val)

    # Convenience properties for the common cross-capability knobs.
    @property
    def output_dir(self) -> Path:
        return Path(self.get_str("output_dir", "outputs")).expanduser()

    @property
    def dtype(self) -> str:
        return self.get_str("dtype", "bfloat16")

    @property
    def offload(self) -> str:
        return self.get_str("offload", "none")

    @property
    def attention(self) -> str:
        return self.get_str("attention", "sdpa")

    @property
    def batch(self) -> int:
        return self.get_int("batch", 1) or 1

    @property
    def seed(self) -> Optional[int]:
        return self.get_int("seed", None)


def _find_config_file(config_path: Optional[str]) -> Optional[Path]:
    if config_path:
        p = Path(config_path).expanduser()
        if not p.exists():
            raise InvalidArgumentError(
                f"config file not found: {p}", remedy="check the --config path"
            )
        return p
    env_path = os.environ.get("LOCALAI_CONFIG")
    if env_path:
        p = Path(env_path).expanduser()
        if p.exists():
            return p
    for name in _DEFAULT_CONFIG_NAMES:
        candidate = Path.cwd() / name
        if candidate.exists():
            return candidate
    return None


def _load_config_file(path: Path) -> Dict[str, Any]:
    try:
        with path.open("rb") as fh:
            return tomllib.load(fh)
    except tomllib.TOMLDecodeError as exc:
        raise InvalidArgumentError(f"invalid TOML in {path}: {exc}")


def _file_layers(data: Dict[str, Any], capability_id: str, model_id: str) -> List[Dict[str, Any]]:
    """Return file-derived layers in increasing precedence order."""
    layers: List[Dict[str, Any]] = []
    # global [defaults]
    if isinstance(data.get("defaults"), dict):
        layers.append(dict(data["defaults"]))
    cap = data.get(capability_id)
    if isinstance(cap, dict):
        # per-capability scalar keys (skip the nested 'models' table)
        layers.append({k: v for k, v in cap.items() if k != "models"})
        models = cap.get("models")
        if isinstance(models, dict) and isinstance(models.get(model_id), dict):
            layers.append(dict(models[model_id]))
    return layers


def _env_layer() -> Dict[str, Any]:
    layer: Dict[str, Any] = {}
    for key in KEY_TYPES:
        env_name = f"LOCALAI_{key.upper()}"
        if env_name in os.environ:
            layer[key] = _coerce(key, os.environ[env_name])
    return layer


def _model_default_layer(spec: ModelSpec) -> Dict[str, Any]:
    # NOTE: width/height are intentionally NOT defaulted here. The capability's
    # size resolver owns the model default so that a --preset (or config preset)
    # is honored instead of being shadowed by a width/height default.
    return {
        "model": spec.model_id,
        "steps": spec.default_steps,
        "guidance": spec.default_guidance,
        "max_sequence_length": spec.max_sequence_length,
        "dtype": spec.recommended_dtype,
    }


def load_settings(
    capability_id: str,
    model_id: str,
    cli_overrides: Optional[Dict[str, Any]] = None,
    config_path: Optional[str] = None,
    spec: Optional[ModelSpec] = None,
    capability_defaults: Optional[Dict[str, Any]] = None,
) -> Settings:
    """Resolve effective :class:`Settings` for ``(capability_id, model_id)``."""
    resolved: Dict[str, Any] = {}

    def apply(layer: Optional[Dict[str, Any]]) -> None:
        if not layer:
            return
        for key, value in layer.items():
            if value is not None:
                resolved[key] = _coerce(key, value)

    # 1-3: built-in defaults (core < capability < model-spec)
    apply(CORE_DEFAULTS)
    apply(capability_defaults)
    if spec is not None:
        apply(_model_default_layer(spec))

    # 4-6: config file layers (global < capability < model)
    cfg_file = _find_config_file(config_path)
    if cfg_file is not None:
        data = _load_config_file(cfg_file)
        for layer in _file_layers(data, capability_id, model_id):
            apply(layer)

    # 7: environment variables
    apply(_env_layer())

    # 8: CLI overrides (highest) — only keys explicitly provided (non-None)
    apply(cli_overrides)

    return Settings(capability_id=capability_id, model_id=model_id, values=resolved)
