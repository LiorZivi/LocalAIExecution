"""The text-to-image FLUX model specs register and resolve correctly."""

from __future__ import annotations

from localai.core import registry


def test_schnell_and_dev_resolve():
    registry.discover_capabilities()
    schnell = registry.get_model("text-to-image", "schnell")
    dev = registry.get_model("text-to-image", "dev")

    assert schnell.repo == "black-forest-labs/FLUX.1-schnell"
    assert schnell.is_default is True
    assert schnell.gated is False
    assert schnell.supports_guidance is False
    assert schnell.supports_negative_prompt is False
    assert schnell.max_sequence_length == 256

    assert dev.repo == "black-forest-labs/FLUX.1-dev"
    assert dev.gated is True
    assert dev.supports_guidance is True
    assert dev.default_guidance == 3.5


def test_default_model_is_schnell():
    registry.discover_capabilities()
    assert registry.default_model("text-to-image").model_id == "schnell"


def test_capability_listed():
    registry.discover_capabilities()
    ids = {c.capability_id for c in registry.list_capabilities()}
    assert "text-to-image" in ids
