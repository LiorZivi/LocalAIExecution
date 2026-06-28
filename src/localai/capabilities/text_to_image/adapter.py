"""The text-to-image capability adapter (FLUX).

Holds all FLUX-specific behaviour: pipeline construction, the schnell-vs-dev
kwarg rules (guidance / negative-prompt / max-sequence-length), seeded
generation, and provenance assembly. Heavy imports (torch/diffusers) are lazy so
registry/CLI discovery stays fast and GPU-free.
"""

from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

from localai.core.errors import (
    CudaUnavailableError,
    GatedModelError,
    NetworkError,
    OutOfMemoryError,
)
from localai.core.interfaces import Artifact, InferenceRequest
from localai.core.metadata import ProvenanceRecord, utc_now_iso
from localai.core.registry import ModelSpec, register_capability

from localai.capabilities.text_to_image import writer as _writer  # noqa: F401 (registers writer)
from localai.capabilities.text_to_image.models import CAPABILITY_ID, MODELS
from localai.capabilities.text_to_image.sizes import resolve_size

_MAX_SEED = 2**32 - 1


@dataclass
class TextToImageRequest(InferenceRequest):
    """Typed request for a FLUX generation."""

    prompt: str = ""
    negative_prompt: Optional[str] = None
    steps: int = 4
    width: int = 1024
    height: int = 1024
    guidance: float = 0.0
    max_sequence_length: int = 256


class TextToImageAdapter:
    """The first concrete capability: prompt -> PNG via FLUX."""

    capability_id = CAPABILITY_ID
    display_name = "Text to Image (FLUX)"
    # FLUX's full bf16 footprint (~33 GB) oversubscribes a 32 GB card with
    # offload=none (spills to shared memory, ~10x slower). Model CPU offload
    # keeps peak VRAM ~24 GB and runs fast; it is the right default here.
    capability_defaults = {"offload": "model"}

    def list_models(self) -> List[ModelSpec]:
        return list(MODELS)

    def register_cli(self, subparsers: Any, shared_parents: List[Any]) -> None:
        from localai.capabilities.text_to_image import cli as t2i_cli

        t2i_cli.register(self, subparsers, shared_parents)

    # ------------------------------------------------------------------ request
    def build_request(self, model_spec: ModelSpec, settings: Any) -> TextToImageRequest:
        prompt = settings.get_str("prompt") if hasattr(settings, "get_str") else None
        if not prompt:
            from localai.core.errors import InvalidArgumentError

            raise InvalidArgumentError("a prompt is required", remedy="provide a prompt")

        width, height = resolve_size(
            settings.get_str("preset"),
            settings.get_int("width"),
            settings.get_int("height"),
            model_spec,
        )
        steps = settings.get_int("steps", model_spec.default_steps)
        if steps is None:
            steps = model_spec.default_steps
        if steps < 1 or steps > 200:
            from localai.core.errors import InvalidArgumentError

            raise InvalidArgumentError(
                f"invalid --steps {steps} (must be between 1 and 200)",
                remedy=f"{model_spec.model_id} works best around {model_spec.default_steps} steps",
            )
        guidance = (
            settings.get_float("guidance", model_spec.default_guidance)
            if model_spec.supports_guidance
            else 0.0
        )
        negative = settings.get_str("negative_prompt") if model_spec.supports_negative_prompt else None
        max_seq = settings.get_int("max_sequence_length", model_spec.max_sequence_length)

        return TextToImageRequest(
            capability_id=model_spec.capability_id,
            model_id=model_spec.model_id,
            seed=settings.seed,
            batch=max(1, settings.batch),
            prompt=prompt,
            negative_prompt=negative,
            steps=int(steps),
            width=int(width),
            height=int(height),
            guidance=float(guidance),
            max_sequence_length=int(max_seq),
        )

    # ------------------------------------------------------------------ loading
    def load_pipeline(
        self, model_spec: ModelSpec, device: str, dtype: Any, offload: str
    ) -> Any:
        # No CPU fallback: confirm CUDA actually drives this GPU first.
        from localai.core.gpu import verify_cuda

        verify_cuda()

        try:
            import torch  # noqa: F401
            from diffusers import FluxPipeline
        except Exception as exc:  # noqa: BLE001
            raise CudaUnavailableError(
                f"failed to import the diffusion stack: {exc}",
                remedy="run scripts\\bootstrap.ps1 to install torch + diffusers",
            )

        token = _resolve_hf_token()
        try:
            pipeline = FluxPipeline.from_pretrained(
                model_spec.repo,
                torch_dtype=dtype,
                token=token,
            )
        except Exception as exc:  # noqa: BLE001
            raise _map_load_error(exc, model_spec)

        _apply_offload(pipeline, offload, device)
        return pipeline

    # ------------------------------------------------------------------ running
    def run(
        self, pipeline: Any, request: InferenceRequest
    ) -> Tuple[List[Artifact], ProvenanceRecord]:
        import torch

        from localai.core.registry import get_model

        assert isinstance(request, TextToImageRequest)
        spec = get_model(request.capability_id, request.model_id)

        seed = request.seed if request.seed is not None else random.randint(0, _MAX_SEED)
        gen_device = "cuda" if torch.cuda.is_available() else "cpu"
        generator = torch.Generator(device=gen_device).manual_seed(int(seed))

        kwargs: dict[str, Any] = {
            "prompt": request.prompt,
            "num_inference_steps": request.steps,
            "width": request.width,
            "height": request.height,
            "max_sequence_length": request.max_sequence_length,
            "num_images_per_prompt": max(1, request.batch),
            "generator": generator,
        }
        if spec.supports_guidance:
            kwargs["guidance_scale"] = request.guidance
        else:
            kwargs["guidance_scale"] = 0.0
        if spec.supports_negative_prompt and request.negative_prompt:
            # FLUX negative prompts require real CFG (doubles compute).
            kwargs["negative_prompt"] = request.negative_prompt
            kwargs["true_cfg_scale"] = max(2.0, request.guidance)

        start = time.perf_counter()
        result = pipeline(**kwargs)
        try:
            torch.cuda.synchronize()
        except Exception:  # pragma: no cover
            pass
        generate_seconds = round(time.perf_counter() - start, 3)

        images = result.images
        params = {
            "prompt": request.prompt,
            "negative_prompt": request.negative_prompt,
            "steps": request.steps,
            "guidance": kwargs["guidance_scale"],
            "width": request.width,
            "height": request.height,
            "max_sequence_length": request.max_sequence_length,
            "batch": len(images),
        }
        record = ProvenanceRecord(
            capability_id=request.capability_id,
            model_id=request.model_id,
            model_repo=spec.repo,
            seed=int(seed),
            timestamp=utc_now_iso(),
            generate_seconds=generate_seconds,
            params=params,
        )
        artifacts = [
            Artifact(type="image", payload=img, suggested_slug=request.prompt)
            for img in images
        ]
        return artifacts, record


# ----------------------------------------------------------------------- helpers
def _resolve_hf_token() -> Optional[str]:
    for env in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACEHUB_API_TOKEN"):
        token = os.environ.get(env)
        if token:
            return token
    # Fall back to the huggingface CLI cache (returns None if absent). The
    # modern API is huggingface_hub.get_token(); HfFolder is the older path.
    try:
        from huggingface_hub import get_token

        return get_token()
    except Exception:  # pragma: no cover
        try:
            from huggingface_hub import HfFolder

            return HfFolder.get_token()
        except Exception:
            return None


def _apply_offload(pipeline: Any, offload: str, device: str) -> None:
    mode = (offload or "none").lower()
    try:
        if mode == "sequential":
            pipeline.enable_sequential_cpu_offload()
        elif mode == "model":
            pipeline.enable_model_cpu_offload()
        else:
            pipeline.to(device)
        # VAE tiling is cheap insurance against decode-time OOM at large sizes.
        if hasattr(pipeline, "vae") and hasattr(pipeline.vae, "enable_tiling"):
            pipeline.vae.enable_tiling()
        elif hasattr(pipeline, "enable_vae_tiling"):
            pipeline.enable_vae_tiling()
    except Exception as exc:  # noqa: BLE001
        if "out of memory" in str(exc).lower():
            raise OutOfMemoryError(
                f"CUDA out of memory placing the pipeline: {exc}",
                remedy="use --offload model or --offload sequential, or reduce size",
            )
        raise


def _map_load_error(exc: Exception, model_spec: ModelSpec) -> Exception:
    """Map upstream load failures to typed errors with actionable remedies.

    Classify by the *actual* failure first (gated markers -> network -> OOM), and
    only fall back to a gated error for a gated model when nothing else matched —
    so an OOM or network blip on a gated model still reports its true exit code.
    """
    name = type(exc).__name__.lower()
    text = str(exc).lower()

    gated_markers = (
        "gatedrepoerror",
        "401",
        "403",
        "awaiting a review",
        "access to model",
        "restricted",
        "must have access",
        "must be authenticated",
    )
    network_markers = (
        "connectionerror",
        "timeout",
        "temporarily",
        "max retries",
        "failed to resolve",
        "name resolution",
        "connection aborted",
    )

    if any(m in text or m in name for m in gated_markers):
        return GatedModelError(
            f"access to gated model '{model_spec.repo}' was denied: {exc}",
            remedy=(
                f"accept the license at https://huggingface.co/{model_spec.repo} "
                "and set HF_TOKEN (or run: hf auth login) with a valid token"
            ),
        )

    if any(m in text or m in name for m in network_markers):
        return NetworkError(
            f"failed to download '{model_spec.repo}': {exc}",
            remedy="check your network; after one download the model works offline",
        )

    if "out of memory" in text:
        return OutOfMemoryError(
            f"CUDA out of memory loading '{model_spec.repo}': {exc}",
            remedy="use --offload model|sequential or reduce size/batch",
        )

    # Unclassified failure of a gated model is most likely an access problem.
    if model_spec.gated:
        return GatedModelError(
            f"could not load gated model '{model_spec.repo}': {exc}",
            remedy=(
                f"accept the license at https://huggingface.co/{model_spec.repo} "
                "and set HF_TOKEN (or run: hf auth login) with a valid token"
            ),
        )
    return exc


# Self-register at import time (the manifest imports this package).
register_capability(TextToImageAdapter())
