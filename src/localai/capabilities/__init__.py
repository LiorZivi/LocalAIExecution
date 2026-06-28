"""Capability registration manifest.

Importing this package self-registers every built-in capability adapter with the
core registry. **Adding a new capability is one import line here** plus the new
adapter module — no edits to ``localai.core``.
"""

from __future__ import annotations

# Each import registers the capability's adapter as a side effect.
from localai.capabilities import text_to_image  # noqa: F401

__all__ = ["text_to_image"]
