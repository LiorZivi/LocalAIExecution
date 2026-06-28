"""Generic output: filename uniqueness, metadata round-trip, writer selection."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from localai.core.config import Settings
from localai.core.errors import InvalidArgumentError
from localai.core.interfaces import Artifact
from localai.core.metadata import ProvenanceRecord
from localai.core.output import build_filename, get_writer, slugify, write_artifact


def _record() -> ProvenanceRecord:
    return ProvenanceRecord(
        capability_id="cap", model_id="mod", model_repo="r/mod", seed=42
    )


def test_filename_unique_same_second(tmp_path):
    rec = _record()
    p1 = build_filename(tmp_path, rec, ext="png")
    p1.write_text("x", encoding="utf-8")
    p2 = build_filename(tmp_path, rec, ext="png")
    p2.write_text("x", encoding="utf-8")
    p3 = build_filename(tmp_path, rec, ext="png")
    assert p1 != p2 != p3
    assert len({p1, p2, p3}) == 3


def test_metadata_round_trip():
    rec = _record()
    rec.params = {"prompt": "hello", "steps": 4}
    rec.generate_seconds = 1.25
    data = json.loads(rec.to_json())
    rec2 = ProvenanceRecord.from_dict(data)
    assert rec2.seed == 42
    assert rec2.params["prompt"] == "hello"
    assert rec2.generate_seconds == 1.25


def test_unknown_writer_raises():
    with pytest.raises(InvalidArgumentError):
        get_writer("no-such-type")


def test_slugify():
    assert slugify("A Cat!! on @ Mat") == "a-cat-on-mat"
    assert slugify("") == "artifact"
    assert len(slugify("x" * 100)) <= 40


def test_write_artifact_with_registered_writer(tmp_path):
    # the 'dummytext' writer is registered by tests/conftest.py
    settings = Settings("cap", "mod", {"output_dir": str(tmp_path)})
    rec = _record()
    art = Artifact(type="dummytext", payload="hello world", suggested_slug="hi there")
    out = write_artifact(art, rec, settings, index=0)
    assert Path(out["path"]).exists()
    assert Path(out["sidecar"]).exists()
    assert out["path"].endswith(".txt")
    side = json.loads(Path(out["sidecar"]).read_text(encoding="utf-8"))
    assert side["seed"] == 42
