"""The common platform/adapter contract.

Every capability (text-to-image today; audio/language/etc. later) implements the
small :class:`CapabilityAdapter` interface. The core engine, CLI dispatcher and
output layer are written against *only* this contract, so a new capability is a
new adapter module plus one registration line — with no edits to the core.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Protocol, Tuple, runtime_checkable

from localai.core.metadata import ProvenanceRecord

if TYPE_CHECKING:  # avoid import cycle: registry imports this module's types
    import argparse

    from localai.core.config import Settings
    from localai.core.registry import ModelSpec


@dataclass
class Artifact:
    """An in-memory generation result plus the type tag used to pick a writer.

    ``payload`` is intentionally untyped (a PIL image today, audio/array/text
    tomorrow). ``suggested_slug`` lets a capability hint at a filename stem.
    """

    type: str
    payload: Any
    suggested_slug: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InferenceRequest:
    """Base request handed to ``adapter.run``.

    Capabilities subclass this to add their own typed fields (prompt, steps,
    size, guidance, ...) but the engine only relies on the base shape.
    """

    capability_id: str
    model_id: str
    seed: Optional[int] = None
    batch: int = 1
    params: Dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class CapabilityAdapter(Protocol):
    """Structural contract every capability must satisfy.

    ``runtime_checkable`` lets conformance tests assert an adapter implements
    the full surface via ``isinstance(adapter, CapabilityAdapter)``.
    """

    capability_id: str
    display_name: str

    def list_models(self) -> List["ModelSpec"]:
        """Return the model specs this capability provides."""
        ...

    def register_cli(
        self, subparsers: "argparse._SubParsersAction", shared_parents: List[Any]
    ) -> None:
        """Contribute this capability's subcommand(s) to the top-level parser."""
        ...

    def build_request(
        self, model_spec: "ModelSpec", settings: "Settings"
    ) -> InferenceRequest:
        """Translate resolved settings into a typed request object."""
        ...

    def load_pipeline(
        self, model_spec: "ModelSpec", device: str, dtype: Any, offload: str
    ) -> Any:
        """Construct and return the loaded inference pipeline."""
        ...

    def run(
        self, pipeline: Any, request: InferenceRequest
    ) -> Tuple[List[Artifact], ProvenanceRecord]:
        """Run inference, returning artifact(s) and a provenance record."""
        ...


__all__ = [
    "Artifact",
    "InferenceRequest",
    "CapabilityAdapter",
    "ProvenanceRecord",
]
