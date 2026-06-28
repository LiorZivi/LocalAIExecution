"""The resident engine loads once, reuses, and unloads (GPU-free dummy)."""

from __future__ import annotations

from localai.core import registry
from localai.core.config import load_settings
from localai.core.engine import Engine


def test_load_once_reused_then_unload(dummy_adapter):
    engine = Engine(device="cpu")
    settings = load_settings("dummy", "m1", spec=registry.get_model("dummy", "m1"))

    p1 = engine.load("dummy", "m1", settings)
    p2 = engine.load("dummy", "m1", settings)
    assert p1 is p2
    assert dummy_adapter.load_count == 1

    req = dummy_adapter.build_request(registry.get_model("dummy", "m1"), settings)
    arts1, rec1 = engine.run(req)
    arts2, rec2 = engine.run(req)
    assert dummy_adapter.load_count == 1  # still one load across two runs
    assert arts1[0].type == "dummytext"
    # engine augments provenance with the runtime fields it owns
    assert rec1.device == "cpu"
    assert rec1.load_seconds is not None

    assert engine.unload("dummy", "m1") is True
    assert engine.is_loaded("dummy", "m1") is False

    engine.load("dummy", "m1", settings)
    assert dummy_adapter.load_count == 2  # reload after unload


def test_run_without_load_raises(dummy_adapter):
    import pytest

    from localai.core.errors import LocalAIError

    engine = Engine(device="cpu")
    settings = load_settings("dummy", "m1", spec=registry.get_model("dummy", "m1"))
    req = dummy_adapter.build_request(registry.get_model("dummy", "m1"), settings)
    with pytest.raises(LocalAIError):
        engine.run(req)
