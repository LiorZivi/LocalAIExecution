"""The PNG image writer embeds params and round-trips through the sidecar."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

import localai.capabilities.text_to_image.writer  # noqa: F401 (registers writer)
from localai.core.config import Settings
from localai.core.interfaces import Artifact
from localai.core.metadata import ProvenanceRecord
from localai.core.output import write_artifact


def test_image_round_trip_and_slug(tmp_path):
    img = Image.new("RGB", (16, 16), (10, 20, 30))
    rec = ProvenanceRecord(
        capability_id="text-to-image",
        model_id="schnell",
        model_repo="black-forest-labs/FLUX.1-schnell",
        seed=7,
        params={"prompt": "a red fox", "steps": 4, "width": 16, "height": 16},
    )
    art = Artifact(type="image", payload=img, suggested_slug="A Red Fox!!")
    settings = Settings("text-to-image", "schnell", {"output_dir": str(tmp_path)})

    out = write_artifact(art, rec, settings, index=0)
    path = Path(out["path"])

    assert path.exists()
    assert path.suffix == ".png"
    assert "a-red-fox" in path.name  # slug sanitized

    reread = Image.open(path)
    assert "localai-params" in reread.text
    assert json.loads(reread.text["localai-params"])["steps"] == 4
    assert reread.text["localai-model"] == "schnell"

    sidecar = json.loads(Path(out["sidecar"]).read_text(encoding="utf-8"))
    assert sidecar["seed"] == 7
    assert sidecar["params"]["prompt"] == "a red fox"
