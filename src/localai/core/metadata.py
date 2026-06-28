"""Modality-agnostic provenance/metadata record.

A :class:`ProvenanceRecord` captures everything needed to understand and
reproduce a generation: which capability + model ran, the seed, timings, the
device/precision, the exact library versions, and a nested capability-specific
``params`` block. It is serialized to the sidecar JSON written next to every
artifact.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional


def utc_now_iso() -> str:
    """Current UTC time as an ISO-8601 string (second precision)."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def collect_library_versions() -> Dict[str, str]:
    """Best-effort versions of the runtime stack, for the provenance record.

    Imports are local and guarded so this works before torch/diffusers exist
    (e.g. in pure-Python unit tests).
    """
    from localai import __version__ as tool_version

    versions: Dict[str, str] = {"localai": tool_version}
    for name in ("torch", "diffusers", "transformers"):
        try:
            module = __import__(name)
            versions[name] = getattr(module, "__version__", "unknown")
        except Exception:  # pragma: no cover - optional at record time
            versions[name] = "not-installed"
    return versions


@dataclass
class ProvenanceRecord:
    """Serializable record describing a single generation run."""

    capability_id: str
    model_id: str
    model_repo: str
    seed: Optional[int] = None
    timestamp: str = field(default_factory=utc_now_iso)
    load_seconds: Optional[float] = None
    generate_seconds: Optional[float] = None
    device: Optional[str] = None
    dtype: Optional[str] = None
    offload: Optional[str] = None
    library_versions: Dict[str, str] = field(default_factory=dict)
    # Capability-specific block (prompt, steps, size, guidance, ...).
    params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=False, default=str)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProvenanceRecord":
        fields = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in fields})
