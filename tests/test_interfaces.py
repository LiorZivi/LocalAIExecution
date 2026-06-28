"""Adapter-interface conformance against the runtime-checkable protocol."""

from __future__ import annotations

from typing import Any, List

from localai.core.interfaces import CapabilityAdapter


class _Complete:
    capability_id = "x"
    display_name = "X"

    def list_models(self) -> List[Any]:
        return []

    def register_cli(self, subparsers: Any, shared_parents: Any) -> None:
        ...

    def build_request(self, model_spec: Any, settings: Any) -> Any:
        ...

    def load_pipeline(self, model_spec: Any, device: str, dtype: Any, offload: str) -> Any:
        ...

    def run(self, pipeline: Any, request: Any) -> Any:
        ...


class _Incomplete:
    capability_id = "y"
    display_name = "Y"

    def list_models(self) -> List[Any]:
        return []


def test_complete_adapter_satisfies_protocol():
    assert isinstance(_Complete(), CapabilityAdapter)


def test_incomplete_adapter_does_not_satisfy_protocol():
    assert not isinstance(_Incomplete(), CapabilityAdapter)


def test_real_text_to_image_adapter_conforms():
    from localai.capabilities.text_to_image.adapter import TextToImageAdapter

    assert isinstance(TextToImageAdapter(), CapabilityAdapter)
