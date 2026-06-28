"""Registry resolves by (capability, model); unknown ids raise typed errors."""

from __future__ import annotations

import pytest

from localai.core import registry
from localai.core.errors import UnknownModelError


def test_resolve_dummy_by_capability_and_model(dummy_adapter):
    spec = registry.get_model("dummy", "m1")
    assert spec.repo == "dummy/m1"
    assert registry.default_model("dummy").model_id == "m1"


def test_list_models(dummy_adapter):
    assert {s.model_id for s in registry.list_models("dummy")} == {"m1", "m2"}


def test_unknown_model_raises(dummy_adapter):
    with pytest.raises(UnknownModelError):
        registry.get_model("dummy", "nope")


def test_unknown_capability_raises():
    with pytest.raises(UnknownModelError):
        registry.get_capability("no-such-capability")


def test_extensibility_no_core_change(dummy_adapter):
    """Registering a brand-new capability needed only an adapter object."""
    assert "dummy" in {c.capability_id for c in registry.list_capabilities()}
