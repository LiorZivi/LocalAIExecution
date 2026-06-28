"""localai.core — the reusable, modality-agnostic runtime.

Owns everything shared across capabilities: GPU bootstrap/verification, the
capability/model registry, layered config, artifact/provenance output, the
resident engine, typed errors, and the CLI dispatcher.
"""

from __future__ import annotations

__all__ = [
    "cli",
    "config",
    "engine",
    "errors",
    "gpu",
    "interfaces",
    "metadata",
    "output",
    "registry",
]
