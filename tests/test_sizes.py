"""Size presets and FLUX dimension validation."""

from __future__ import annotations

import pytest

from localai.capabilities.image.text_to_image.models import SCHNELL
from localai.capabilities.image.text_to_image.sizes import PRESETS, resolve_size
from localai.core.errors import InvalidArgumentError


def test_each_preset_maps_to_expected_dims():
    for name, (w, h) in PRESETS.items():
        assert resolve_size(name, None, None, SCHNELL) == (w, h)


def test_default_when_nothing_given():
    assert resolve_size(None, None, None, SCHNELL) == (
        SCHNELL.default_width,
        SCHNELL.default_height,
    )


def test_explicit_valid_dims():
    assert resolve_size(None, 512, 768, SCHNELL) == (512, 768)


def test_explicit_overrides_preset():
    assert resolve_size("square", 768, 512, SCHNELL) == (768, 512)


def test_non_multiple_rejected():
    with pytest.raises(InvalidArgumentError):
        resolve_size(None, 500, 512, SCHNELL)


def test_single_dimension_rejected():
    with pytest.raises(InvalidArgumentError):
        resolve_size(None, 512, None, SCHNELL)


def test_unknown_preset_rejected():
    with pytest.raises(InvalidArgumentError):
        resolve_size("triangle", None, None, SCHNELL)


def test_out_of_range_rejected():
    with pytest.raises(InvalidArgumentError):
        resolve_size(None, 64, 64, SCHNELL)


def test_preset_resolves_through_settings():
    """Regression: a --preset must not be shadowed by model width/height defaults."""
    from localai.capabilities.image.text_to_image.adapter import TextToImageAdapter
    from localai.core.config import load_settings

    spec = SCHNELL
    settings = load_settings(
        "text-to-image",
        "schnell",
        cli_overrides={"prompt": "x", "preset": "widescreen"},
        spec=spec,
    )
    req = TextToImageAdapter().build_request(spec, settings)
    assert (req.width, req.height) == (1344, 768)


def test_default_size_through_settings():
    from localai.capabilities.image.text_to_image.adapter import TextToImageAdapter
    from localai.core.config import load_settings

    settings = load_settings(
        "text-to-image", "schnell", cli_overrides={"prompt": "x"}, spec=SCHNELL
    )
    req = TextToImageAdapter().build_request(SCHNELL, settings)
    assert (req.width, req.height) == (1024, 1024)
