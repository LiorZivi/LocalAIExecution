"""Text-to-image capability (FLUX) — the first and only capability built now.

Importing this package registers the :class:`TextToImageAdapter` with the core
registry (via ``adapter`` import side effect) and the PNG image writer.
"""

from __future__ import annotations

# Importing the adapter self-registers the capability + its image writer.
from localai.capabilities.image.text_to_image import adapter  # noqa: F401

__all__ = ["adapter"]
