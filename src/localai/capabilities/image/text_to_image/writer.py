"""Concrete PNG image writer for artifact type ``image``.

Encodes the PIL image to PNG with the provenance ``params`` embedded as PNG text
chunks. Registered into the core writer-selection map on import; the core writes
the ``.json`` sidecar companion.
"""

from __future__ import annotations

import json
from pathlib import Path

from localai.core.interfaces import Artifact
from localai.core.metadata import ProvenanceRecord
from localai.core.output import register_writer

ARTIFACT_TYPE = "image"


def image_writer(artifact: Artifact, path: Path, record: ProvenanceRecord) -> None:
    """Write the PIL image at *path* as PNG with embedded provenance text."""
    from PIL.PngImagePlugin import PngInfo  # local import keeps discovery light

    image = artifact.payload
    info = PngInfo()
    info.add_text("localai-capability", str(record.capability_id))
    info.add_text("localai-model", str(record.model_id))
    info.add_text("localai-model-repo", str(record.model_repo))
    if record.seed is not None:
        info.add_text("localai-seed", str(record.seed))
    prompt = record.params.get("prompt")
    if prompt:
        info.add_text("parameters", str(prompt))
    info.add_text("localai-params", json.dumps(record.params, default=str))
    image.save(path, format="PNG", pnginfo=info)


# Register on import (the adapter imports this module at capability load time).
register_writer(ARTIFACT_TYPE, image_writer, "png")
