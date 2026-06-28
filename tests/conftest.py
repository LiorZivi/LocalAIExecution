"""Shared test fixtures: a GPU-free dummy capability that exercises the core.

The dummy adapter proves the core (registry, engine, output, CLI dispatch,
``--json``) carries no image-specific assumptions — it runs without torch/CUDA.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List

import pytest

from localai.core import registry
from localai.core.config import load_settings
from localai.core.engine import Engine
from localai.core.interfaces import Artifact, InferenceRequest
from localai.core.metadata import ProvenanceRecord
from localai.core.output import register_writer, write_artifact
from localai.core.registry import ModelSpec


def _dummy_writer(artifact: Artifact, path: Path, record: ProvenanceRecord) -> None:
    path.write_text(str(artifact.payload), encoding="utf-8")


register_writer("dummytext", _dummy_writer, "txt")


class DummyAdapter:
    """A minimal capability used to test the core without a GPU."""

    capability_id = "dummy"
    display_name = "Dummy Capability"

    def __init__(self) -> None:
        self.load_count = 0

    def list_models(self) -> List[ModelSpec]:
        return [
            ModelSpec(
                model_id="m1",
                capability_id="dummy",
                repo="dummy/m1",
                pipeline_class="Fake",
                default_steps=2,
                is_default=True,
            ),
            ModelSpec(
                model_id="m2",
                capability_id="dummy",
                repo="dummy/m2",
                pipeline_class="Fake",
                default_steps=5,
            ),
        ]

    def register_cli(self, subparsers: Any, shared_parents: List[Any]) -> None:
        p = subparsers.add_parser("dummy-run", parents=shared_parents, help="dummy run")
        p.add_argument("text")
        p.set_defaults(func=self._handler)

    def build_request(self, model_spec: ModelSpec, settings: Any) -> InferenceRequest:
        return InferenceRequest(
            capability_id="dummy",
            model_id=model_spec.model_id,
            seed=settings.seed,
            params={"text": settings.get("text", "hi"), "steps": settings.get_int("steps")},
        )

    def load_pipeline(self, model_spec: ModelSpec, device: str, dtype: Any, offload: str) -> Any:
        self.load_count += 1
        return {"fake": True, "device": device, "model": model_spec.model_id}

    def run(self, pipeline: Any, request: InferenceRequest):
        record = ProvenanceRecord(
            capability_id="dummy",
            model_id=request.model_id,
            model_repo=f"dummy/{request.model_id}",
            seed=request.seed,
            generate_seconds=0.001,
            params=dict(request.params),
        )
        artifact = Artifact(
            type="dummytext", payload=request.params.get("text", "hi"), suggested_slug="dummy"
        )
        return [artifact], record

    def _handler(self, args: Any) -> int:
        from localai.core.cli import emit_result

        model_id = getattr(args, "model", None) or "m1"
        overrides: dict = {"text": args.text}
        for key in ("output_dir", "seed", "steps"):
            value = getattr(args, key, None)
            if value is not None:
                overrides[key] = value
        settings = load_settings(
            "dummy", model_id, cli_overrides=overrides, spec=registry.get_model("dummy", model_id)
        )
        engine = Engine(device="cpu")
        engine.load("dummy", settings.model_id, settings)
        artifacts, record = engine.run(
            self.build_request(registry.get_model("dummy", settings.model_id), settings)
        )
        written = [write_artifact(a, record, settings, index=i) for i, a in enumerate(artifacts)]
        emit_result(bool(getattr(args, "json_mode", False)), "dummy", settings.model_id, written)
        return 0


@pytest.fixture
def dummy_adapter():
    """Register the dummy capability additively; clean up afterwards."""
    registry.discover_capabilities()  # ensure real capabilities are loaded once
    adapter = DummyAdapter()
    registry.register_capability(adapter)
    try:
        yield adapter
    finally:
        registry.unregister_capability("dummy")
