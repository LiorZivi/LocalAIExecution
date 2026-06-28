"""The resident engine: load a pipeline once, reuse it across runs.

The engine is modality-agnostic. It owns device/dtype/offload selection and a
pipeline cache keyed by ``(capability_id, model_id)``, delegating all model
specifics to the owning adapter's ``load_pipeline`` / ``run``. It augments the
adapter's provenance record with the shared runtime fields it owns.
"""

from __future__ import annotations

import gc
import time
from typing import Any, Dict, Optional, Tuple

from localai.core import registry
from localai.core.errors import LocalAIError, OutOfMemoryError
from localai.core.interfaces import Artifact, InferenceRequest
from localai.core.metadata import ProvenanceRecord, collect_library_versions

_DTYPE_ALIASES = {
    "bf16": "bfloat16",
    "fp16": "float16",
    "half": "float16",
    "fp32": "float32",
    "float": "float32",
}


def _resolve_torch_dtype(dtype_name: str) -> Any:
    """Map a dtype string to a torch dtype, or return the string if torch absent."""
    name = _DTYPE_ALIASES.get(dtype_name.lower(), dtype_name.lower())
    try:
        import torch
    except Exception:  # pragma: no cover - torch optional in pure tests
        return name
    return getattr(torch, name, torch.bfloat16)


def _is_cuda_oom(exc: BaseException) -> bool:
    try:
        import torch

        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
    except Exception:  # pragma: no cover
        pass
    return "out of memory" in str(exc).lower()


class Engine:
    """Loads pipelines once and reuses them; frees VRAM on unload."""

    def __init__(self, device: Optional[str] = None) -> None:
        self._device = device
        self._cache: Dict[Tuple[str, str], Any] = {}
        self._meta: Dict[Tuple[str, str], Dict[str, Any]] = {}

    def resolve_device(self) -> str:
        if self._device:
            return self._device
        try:
            import torch

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:  # pragma: no cover - torch optional
            self._device = "cpu"
        return self._device

    def is_loaded(self, capability_id: str, model_id: str) -> bool:
        return (capability_id, model_id) in self._cache

    def load(self, capability_id: str, model_id: str, settings: Any) -> Any:
        """Load (or return cached) pipeline for ``(capability_id, model_id)``."""
        key = (capability_id, model_id)
        if key in self._cache:
            return self._cache[key]

        adapter = registry.get_capability(capability_id)
        spec = registry.get_model(capability_id, model_id)
        device = self.resolve_device()
        dtype_name = getattr(settings, "dtype", None) or spec.recommended_dtype
        offload = getattr(settings, "offload", None) or "none"
        dtype = _resolve_torch_dtype(dtype_name)

        start = time.perf_counter()
        try:
            pipeline = adapter.load_pipeline(spec, device, dtype, offload)
        except LocalAIError:
            raise
        except Exception as exc:  # noqa: BLE001 - map low-level failures
            if _is_cuda_oom(exc):
                raise OutOfMemoryError(
                    f"CUDA out of memory while loading '{model_id}': {exc}",
                    remedy="enable offload (model|sequential), or reduce size/batch",
                )
            raise
        load_seconds = round(time.perf_counter() - start, 3)

        self._cache[key] = pipeline
        self._meta[key] = {
            "device": device,
            "dtype": dtype_name,
            "offload": offload,
            "load_seconds": load_seconds,
        }
        return pipeline

    def run(self, request: InferenceRequest) -> Tuple[list[Artifact], ProvenanceRecord]:
        """Run inference for a previously-loaded model and augment provenance."""
        key = (request.capability_id, request.model_id)
        if key not in self._cache:
            raise LocalAIError(
                f"model '{request.model_id}' for capability "
                f"'{request.capability_id}' is not loaded",
            )
        adapter = registry.get_capability(request.capability_id)
        pipeline = self._cache[key]
        try:
            artifacts, record = adapter.run(pipeline, request)
        except LocalAIError:
            raise
        except Exception as exc:  # noqa: BLE001
            if _is_cuda_oom(exc):
                raise OutOfMemoryError(
                    f"CUDA out of memory during generation: {exc}",
                    remedy="reduce size/batch or enable offload (model|sequential)",
                )
            raise

        meta = self._meta.get(key, {})
        if record.load_seconds is None:
            record.load_seconds = meta.get("load_seconds")
        record.device = record.device or meta.get("device")
        record.dtype = record.dtype or meta.get("dtype")
        record.offload = record.offload or meta.get("offload")
        if not record.library_versions:
            record.library_versions = collect_library_versions()
        return artifacts, record

    def unload(self, capability_id: str, model_id: str) -> bool:
        """Evict a cached pipeline and free CUDA memory. Returns True if evicted."""
        key = (capability_id, model_id)
        pipeline = self._cache.pop(key, None)
        self._meta.pop(key, None)
        if pipeline is None:
            return False
        del pipeline
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception:  # pragma: no cover
            pass
        return True

    def unload_all(self) -> None:
        for cap, model in list(self._cache):
            self.unload(cap, model)
