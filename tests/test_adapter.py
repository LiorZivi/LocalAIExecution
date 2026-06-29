"""Adapter request-building validation (GPU-free)."""

from __future__ import annotations

import pytest

from localai.capabilities.image.text_to_image.adapter import TextToImageAdapter
from localai.capabilities.image.text_to_image.models import DEV, SCHNELL
from localai.core.config import load_settings
from localai.core.errors import InvalidArgumentError


def _settings(overrides):
    return load_settings("text-to-image", "schnell", cli_overrides=overrides, spec=SCHNELL)


def test_prompt_required():
    with pytest.raises(InvalidArgumentError):
        TextToImageAdapter().build_request(SCHNELL, _settings({}))


def test_steps_zero_rejected():
    with pytest.raises(InvalidArgumentError):
        TextToImageAdapter().build_request(SCHNELL, _settings({"prompt": "x", "steps": 0}))


def test_steps_too_high_rejected():
    with pytest.raises(InvalidArgumentError):
        TextToImageAdapter().build_request(SCHNELL, _settings({"prompt": "x", "steps": 999}))


def test_schnell_forces_guidance_zero_and_no_negative():
    req = TextToImageAdapter().build_request(
        SCHNELL, _settings({"prompt": "x", "negative_prompt": "ugly"})
    )
    assert req.guidance == 0.0
    assert req.negative_prompt is None  # schnell ignores negative prompts
    assert req.max_sequence_length == 256


def test_dev_supports_guidance_and_negative():
    settings = load_settings(
        "text-to-image",
        "dev",
        cli_overrides={"prompt": "x", "guidance": 4.0, "negative_prompt": "blurry"},
        spec=DEV,
    )
    req = TextToImageAdapter().build_request(DEV, settings)
    assert req.guidance == 4.0
    assert req.negative_prompt == "blurry"
    assert req.max_sequence_length == 512


def test_default_offload_is_model_for_flux():
    settings = load_settings(
        "text-to-image",
        "schnell",
        cli_overrides={"prompt": "x"},
        spec=SCHNELL,
        capability_defaults=TextToImageAdapter().capability_defaults,
    )
    assert settings.offload == "model"


def test_map_load_error_classifies_before_gated_fallback():
    from localai.capabilities.image.text_to_image.adapter import _map_load_error
    from localai.core.errors import GatedModelError, NetworkError, OutOfMemoryError

    # A gated model must still surface OOM and network with their true codes.
    assert isinstance(
        _map_load_error(RuntimeError("CUDA out of memory"), DEV), OutOfMemoryError
    )
    assert isinstance(
        _map_load_error(RuntimeError("Connection aborted: max retries exceeded"), DEV),
        NetworkError,
    )
    # A real gated denial maps to gated.
    assert isinstance(
        _map_load_error(RuntimeError("401: access to model is restricted"), DEV),
        GatedModelError,
    )
    # An unclear failure of a gated model falls back to gated.
    assert isinstance(_map_load_error(RuntimeError("weird"), DEV), GatedModelError)
    # schnell is login-gated on HF now: a 401 still maps to gated...
    assert isinstance(
        _map_load_error(RuntimeError("401 ... must be authenticated"), SCHNELL),
        GatedModelError,
    )
    # ...but an unclear failure of an ungated model passes through unchanged.
    passthrough = RuntimeError("some unrelated bug")
    assert _map_load_error(passthrough, SCHNELL) is passthrough
