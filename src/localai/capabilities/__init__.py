"""Capability registration manifest.

Importing this package self-registers every built-in capability adapter with the
core registry. Capabilities are grouped by modality (``image``, and future
``video``, ``audio``, ...); each modality package imports its own capabilities.

**Adding a new modality is one import line here** plus the new modality package
(which lists its capabilities) — no edits to ``localai.core``.
"""

from __future__ import annotations

# Each modality group imports its capabilities, registering their adapters.
from localai.capabilities import image  # noqa: F401

__all__ = ["image"]
