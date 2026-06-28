"""Capability/model registry keyed by capability id + model id.

Capabilities self-register on import (via the ``capabilities`` manifest). The
core looks models up by ``(capability_id, model_id)``; unknown ids raise the
typed :class:`~localai.core.errors.UnknownModelError`. A :class:`ModelSpec`
captures the per-model behaviour the engine and adapters need.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from localai.core.errors import UnknownModelError

if TYPE_CHECKING:  # avoid import cycle with interfaces
    from localai.core.interfaces import CapabilityAdapter


@dataclass
class ModelSpec:
    """Static description of a single model within a capability."""

    model_id: str
    capability_id: str
    repo: str
    pipeline_class: str
    display_name: str = ""
    default_steps: int = 4
    min_steps: int = 1
    max_steps: int = 100
    supports_guidance: bool = False
    default_guidance: float = 0.0
    supports_negative_prompt: bool = False
    gated: bool = False
    default_width: int = 1024
    default_height: int = 1024
    size_multiple: int = 16
    max_sequence_length: int = 256
    recommended_dtype: str = "bfloat16"
    is_default: bool = False
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.display_name:
            self.display_name = self.model_id


# Registry state: capability id -> adapter, and (capability, model) -> spec.
_capabilities: Dict[str, "CapabilityAdapter"] = {}
_models: Dict[Tuple[str, str], ModelSpec] = {}
_discovered = False


def register_capability(adapter: "CapabilityAdapter") -> None:
    """Register a capability adapter and index all of its model specs."""
    cap_id = adapter.capability_id
    _capabilities[cap_id] = adapter
    for spec in adapter.list_models():
        _models[(cap_id, spec.model_id)] = spec


def get_capability(capability_id: str) -> "CapabilityAdapter":
    discover_capabilities()
    try:
        return _capabilities[capability_id]
    except KeyError:
        known = ", ".join(sorted(_capabilities)) or "(none)"
        raise UnknownModelError(
            f"unknown capability '{capability_id}'",
            remedy=f"available capabilities: {known}",
        )


def list_capabilities() -> List["CapabilityAdapter"]:
    discover_capabilities()
    return list(_capabilities.values())


def get_model(capability_id: str, model_id: str) -> ModelSpec:
    """Resolve a model spec by ``(capability_id, model_id)``."""
    discover_capabilities()
    # Surface an unknown *capability* before an unknown *model*.
    if capability_id not in _capabilities:
        get_capability(capability_id)
    try:
        return _models[(capability_id, model_id)]
    except KeyError:
        known = ", ".join(sorted(m for (c, m) in _models if c == capability_id)) or "(none)"
        raise UnknownModelError(
            f"unknown model '{model_id}' for capability '{capability_id}'",
            remedy=f"available models: {known}",
        )


def list_models(capability_id: str) -> List[ModelSpec]:
    discover_capabilities()
    get_capability(capability_id)  # validates capability id
    return [spec for (cap, _), spec in _models.items() if cap == capability_id]


def default_model(capability_id: str) -> ModelSpec:
    """Return the capability's default model (``is_default`` or first listed)."""
    specs = list_models(capability_id)
    for spec in specs:
        if spec.is_default:
            return spec
    if specs:
        return specs[0]
    raise UnknownModelError(
        f"capability '{capability_id}' has no models registered",
    )


def discover_capabilities(force: bool = False) -> None:
    """Import the capabilities manifest so adapters self-register.

    Idempotent: the manifest is imported once unless ``force`` is set (used by
    tests that reset registry state).
    """
    global _discovered
    if _discovered and not force:
        return
    _discovered = True
    importlib.import_module("localai.capabilities")


def reset_registry() -> None:
    """Clear all registry state (test helper)."""
    global _discovered
    _capabilities.clear()
    _models.clear()
    _discovered = False


def unregister_capability(capability_id: str) -> None:
    """Remove a capability and its models from the registry (test helper)."""
    _capabilities.pop(capability_id, None)
    for key in [k for k in _models if k[0] == capability_id]:
        _models.pop(key, None)
