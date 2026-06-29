"""Image modality group.

Groups all image-generating capabilities. Today that is ``text_to_image``
(FLUX); future image capabilities (e.g. ``image_to_image``) live here too as
sibling packages. Importing a capability module self-registers its adapter.

**Adding a new image capability is one import line here** plus the new
capability package — no edits to ``localai.core``.
"""

from __future__ import annotations

from localai.capabilities.image import text_to_image  # noqa: F401

__all__ = ["text_to_image"]
