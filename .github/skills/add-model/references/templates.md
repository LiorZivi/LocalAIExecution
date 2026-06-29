# Templates & Skeletons

Copy-shaped starting points for the two cases. They mirror the live
`src\localai\capabilities\image\text_to_image\` package — **treat that package as
the source of truth** and reconcile against it if anything here looks stale.
Capabilities are grouped by modality: `capabilities\<modality>\<capability>\`
(e.g. `image\text_to_image\`). Replace every `<...>` placeholder, including
`<modality>`. Keep heavy imports (`torch`, `diffusers`) lazy.

## Table of contents

- [Case A — add a model variant (a `ModelSpec` + adapter branch points)](#case-a)
- [Case B — a new capability package](#case-b)
  - [`models.py`](#models-py)
  - [`adapter.py`](#adapter-py)
  - [`cli.py`](#cli-py)
  - [`writer.py` (only for a new artifact type)](#writer-py)
  - [the modality + top-level registration manifests](#registration)
  - [tests](#tests)

---

<a id="case-a"></a>
## Case A — add a model variant to an existing capability

Add a `ModelSpec` to the capability's `models.py` and include it in `MODELS`:

```python
NEW_MODEL = ModelSpec(
    model_id="<short-id>",                 # e.g. "dev"
    capability_id=CAPABILITY_ID,
    repo="<org>/<repo>",                   # Hugging Face repo
    pipeline_class="<PipelineClass>",      # e.g. "FluxPipeline"
    display_name="<human label>",
    default_steps=<int>,
    min_steps=<int>,
    max_steps=<int>,
    supports_guidance=<bool>,
    default_guidance=<float>,
    supports_negative_prompt=<bool>,
    gated=<bool>,                          # True if login-gated on HF
    default_width=1024,
    default_height=1024,
    size_multiple=<int>,
    max_sequence_length=<int>,
    recommended_dtype="bfloat16",
    is_default=False,                      # only ONE default per capability
    notes="<anything a user should know>",
)

MODELS = [EXISTING_MODEL, NEW_MODEL]
```

Then branch the adapter **only where the new model genuinely differs**, keyed off
spec flags rather than the model id. The FLUX adapter is the pattern: in `run`,

```python
if spec.supports_guidance:
    kwargs["guidance_scale"] = request.guidance
if spec.supports_negative_prompt and request.negative_prompt:
    kwargs["negative_prompt"] = request.negative_prompt
    kwargs["true_cfg_scale"] = max(2.0, request.guidance)
```

If the new model needs a **different pipeline class**, construct it from
`model_spec.pipeline_class` in `load_pipeline` instead of importing one fixed
class. Avoid `if model_id == "...":` — add a spec field and read it.

---

<a id="case-b"></a>
## Case B — a new capability package

Create `src\localai\capabilities\<modality>\<your_capability>\` with the files
below (e.g. `audio\text_to_speech\`). If `<modality>` is new, you'll also create
its manifest `__init__.py` — see [registration](#registration).

<a id="models-py"></a>
### `models.py`

```python
"""<Your capability> model specifications."""
from __future__ import annotations

from localai.core.registry import ModelSpec

CAPABILITY_ID = "<your-capability-id>"   # e.g. "text-to-speech"

PRIMARY = ModelSpec(
    model_id="<short-id>",
    capability_id=CAPABILITY_ID,
    repo="<org>/<repo>",
    pipeline_class="<PipelineClass>",
    display_name="<human label>",
    gated=<bool>,
    recommended_dtype="bfloat16",
    is_default=True,
    notes="<license / usage notes>",
    # Reuse the generic numeric fields that fit your modality; ignore the rest.
)

MODELS = [PRIMARY]
```

<a id="adapter-py"></a>
### `adapter.py`

```python
"""The <your capability> adapter. Heavy imports stay lazy."""
from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

from localai.core.errors import (
    CudaUnavailableError,
    GatedModelError,
    InvalidArgumentError,
    NetworkError,
    OutOfMemoryError,
)
from localai.core.interfaces import Artifact, InferenceRequest
from localai.core.metadata import ProvenanceRecord, utc_now_iso
from localai.core.registry import ModelSpec, register_capability

from localai.capabilities.<modality>.<your_capability> import writer as _writer  # noqa: F401 (registers writer)
from localai.capabilities.<modality>.<your_capability>.models import CAPABILITY_ID, MODELS

_MAX_SEED = 2**32 - 1


@dataclass
class <Cap>Request(InferenceRequest):
    """Typed inputs for one generation."""
    # add your fields, e.g. prompt: str = ""


class <Cap>Adapter:
    capability_id = CAPABILITY_ID
    display_name = "<Human Name>"
    # Set only if your model's footprint needs CPU offload by default:
    # capability_defaults = {"offload": "model"}

    def list_models(self) -> List[ModelSpec]:
        return list(MODELS)

    def register_cli(self, subparsers: Any, shared_parents: List[Any]) -> None:
        from localai.capabilities.<modality>.<your_capability> import cli as _cli
        _cli.register(self, subparsers, shared_parents)

    def build_request(self, model_spec: ModelSpec, settings: Any) -> "<Cap>Request":
        # Validate cheap args here (before the expensive load). Prefer
        # settings.get_int/get_float/get_str so you don't touch core config.
        # Raise InvalidArgumentError(..., remedy=...) on bad input.
        return <Cap>Request(
            capability_id=model_spec.capability_id,
            model_id=model_spec.model_id,
            seed=settings.seed,
            batch=max(1, settings.batch),
            # ...your fields...
        )

    def load_pipeline(self, model_spec: ModelSpec, device: str, dtype: Any, offload: str) -> Any:
        from localai.core.gpu import verify_cuda
        verify_cuda()  # no silent CPU fallback

        try:
            import torch  # noqa: F401
            from <library> import <PipelineClass>
        except Exception as exc:  # noqa: BLE001
            raise CudaUnavailableError(
                f"failed to import the model stack: {exc}",
                remedy="run scripts\\bootstrap.ps1 to install the dependencies",
            )

        token = _resolve_hf_token()
        try:
            pipeline = <PipelineClass>.from_pretrained(
                model_spec.repo, torch_dtype=dtype, token=token,
            )
        except Exception as exc:  # noqa: BLE001
            raise _map_load_error(exc, model_spec)

        _apply_offload(pipeline, offload, device)
        return pipeline

    def run(self, pipeline: Any, request: InferenceRequest) -> Tuple[List[Artifact], ProvenanceRecord]:
        import torch
        from localai.core.registry import get_model

        assert isinstance(request, <Cap>Request)
        spec = get_model(request.capability_id, request.model_id)

        seed = request.seed if request.seed is not None else random.randint(0, _MAX_SEED)
        gen_device = "cuda" if torch.cuda.is_available() else "cpu"
        generator = torch.Generator(device=gen_device).manual_seed(int(seed))

        start = time.perf_counter()
        result = pipeline(...)  # build kwargs from request; pass generator
        try:
            torch.cuda.synchronize()
        except Exception:  # pragma: no cover
            pass
        generate_seconds = round(time.perf_counter() - start, 3)

        params = { ... }  # capability-specific block for provenance
        record = ProvenanceRecord(
            capability_id=request.capability_id,
            model_id=request.model_id,
            model_repo=spec.repo,
            seed=int(seed),
            timestamp=utc_now_iso(),
            generate_seconds=generate_seconds,
            params=params,
        )
        artifacts = [Artifact(type="<artifact-type>", payload=<payload>, suggested_slug="<slug>")]
        return artifacts, record


# --- helpers: copy these from text_to_image/adapter.py and adapt the markers ---
def _resolve_hf_token() -> Optional[str]:
    import os
    for env in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACEHUB_API_TOKEN"):
        if os.environ.get(env):
            return os.environ[env]
    try:
        from huggingface_hub import get_token
        return get_token()
    except Exception:  # pragma: no cover
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
    except Exception as exc:  # noqa: BLE001
        if "out of memory" in str(exc).lower():
            raise OutOfMemoryError(
                f"CUDA out of memory placing the pipeline: {exc}",
                remedy="use --offload model or --offload sequential",
            )
        raise


def _map_load_error(exc: Exception, model_spec: ModelSpec) -> Exception:
    """Classify by the actual failure: gated -> network -> OOM, then fall back to
    a gated error for a gated model. Copy the marker lists from the FLUX adapter."""
    text = str(exc).lower()
    name = type(exc).__name__.lower()
    if any(m in text or m in name for m in ("gatedrepoerror", "401", "403", "restricted", "must be authenticated")):
        return GatedModelError(
            f"access to gated model '{model_spec.repo}' was denied: {exc}",
            remedy=f"accept the license at https://huggingface.co/{model_spec.repo} and set HF_TOKEN (or run: hf auth login)",
        )
    if any(m in text or m in name for m in ("connectionerror", "timeout", "max retries", "failed to resolve")):
        return NetworkError(f"failed to download '{model_spec.repo}': {exc}",
                            remedy="check your network; after one download it works offline")
    if "out of memory" in text:
        return OutOfMemoryError(f"CUDA out of memory loading '{model_spec.repo}': {exc}",
                                remedy="use --offload model|sequential or reduce size/batch")
    if model_spec.gated:
        return GatedModelError(f"could not load gated model '{model_spec.repo}': {exc}",
                               remedy=f"accept the license at https://huggingface.co/{model_spec.repo} and set HF_TOKEN")
    return exc


register_capability(<Cap>Adapter())  # self-register at import time
```

<a id="cli-py"></a>
### `cli.py`

```python
"""CLI wiring for <your capability>."""
from __future__ import annotations

import argparse
import sys
from typing import Any, Dict, List, Optional

from localai.core import registry
from localai.core.cli import emit_result
from localai.core.config import load_settings
from localai.core.engine import Engine
from localai.core.errors import EXIT_SUCCESS
from localai.core.output import write_artifact

# argparse dests pulled into the settings layer (None values are ignored).
_OVERRIDE_KEYS = ("model", "output_dir", "seed", "dtype", "offload", "batch",
                  # ...your modality-specific dests...)


def register(adapter: Any, subparsers: argparse._SubParsersAction, shared_parents: List[Any]) -> None:
    p = subparsers.add_parser(
        "<your-subcommand>",                 # e.g. "speak"
        parents=shared_parents,              # inherits --json/--model/--seed/... 
        help="<one-line help>",
        description="<longer description>",
    )
    # p.add_argument("prompt", help="...")   # positional inputs
    # p.add_argument("--your-flag", ...)     # modality-specific flags
    p.set_defaults(func=lambda args: _handler(adapter, args))


def _collect_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    return {k: getattr(args, k) for k in _OVERRIDE_KEYS if getattr(args, k, None) is not None}


def _resolve_model_id(capability_id: str, requested: Optional[str]) -> str:
    if requested:
        registry.get_model(capability_id, requested)  # validates / raises
        return requested
    return registry.default_model(capability_id).model_id


def _handler(adapter: Any, args: argparse.Namespace) -> int:
    json_mode = bool(getattr(args, "json_mode", False))
    cap_id = adapter.capability_id
    model_id = _resolve_model_id(cap_id, getattr(args, "model", None))
    spec = registry.get_model(cap_id, model_id)

    settings = load_settings(
        cap_id, model_id,
        cli_overrides=_collect_overrides(args),
        config_path=getattr(args, "config_path", None),
        spec=spec,
        capability_defaults=getattr(adapter, "capability_defaults", None),
    )

    engine = Engine()
    request = adapter.build_request(spec, settings)   # validate cheap args before load
    engine.load(cap_id, model_id, settings)
    artifacts, record = engine.run(request)
    written = [write_artifact(a, record, settings, index=i) for i, a in enumerate(artifacts)]
    emit_result(json_mode, cap_id, model_id, written)
    return EXIT_SUCCESS
```

<a id="writer-py"></a>
### `writer.py` (only if you emit a NEW artifact type)

Reuse the existing `image` writer if you produce PNGs. Otherwise:

```python
"""Concrete writer for artifact type '<artifact-type>'."""
from __future__ import annotations

from pathlib import Path

from localai.core.interfaces import Artifact
from localai.core.metadata import ProvenanceRecord
from localai.core.output import register_writer

ARTIFACT_TYPE = "<artifact-type>"   # e.g. "audio"


def <type>_writer(artifact: Artifact, path: Path, record: ProvenanceRecord) -> None:
    # local imports keep discovery light
    payload = artifact.payload
    # ...encode payload to `path` (e.g. soundfile.write, np.save, text)...


register_writer(ARTIFACT_TYPE, <type>_writer, "<ext>")   # core writes the .json sidecar
```

<a id="registration"></a>
### Registration manifests (the "one line")

Importing a capability module self-registers its adapter, so registration is just
an import in the right manifest. Which manifest depends on whether the modality
already exists.

**The capability's own `__init__.py`** —
`capabilities\<modality>\<your_capability>\__init__.py` — imports the adapter so
the side effect fires:

```python
from localai.capabilities.<modality>.<your_capability> import adapter  # noqa: F401

__all__ = ["adapter"]
```

**Existing modality** (e.g. another capability under `image\`): add one line to
the modality manifest `capabilities\<modality>\__init__.py`:

```python
from localai.capabilities.<modality> import <your_capability>  # noqa: F401

__all__ = ["text_to_image", "<your_capability>"]
```

**New modality** (e.g. `audio\`): create the modality manifest
`capabilities\<modality>\__init__.py` importing your capability (as just above),
then add one line to the top-level `capabilities\__init__.py`:

```python
from localai.capabilities import <modality>  # noqa: F401

__all__ = ["image", "<modality>"]
```

<a id="tests"></a>
### Tests (GPU-free)

```python
"""<your capability> registers and resolves correctly."""
from __future__ import annotations

from localai.core import registry
from localai.core.interfaces import CapabilityAdapter


def test_adapter_conforms():
    from localai.capabilities.<modality>.<your_capability>.adapter import <Cap>Adapter
    assert isinstance(<Cap>Adapter(), CapabilityAdapter)


def test_model_resolves_and_is_listed():
    registry.discover_capabilities()
    spec = registry.get_model("<your-capability-id>", "<short-id>")
    assert spec.repo == "<org>/<repo>"
    assert registry.default_model("<your-capability-id>").model_id == "<short-id>"
    assert "<your-capability-id>" in {c.capability_id for c in registry.list_capabilities()}
```

Add focused unit tests for `build_request` and any input validation, calling them
directly without loading a pipeline. Model the dummy-adapter, no-GPU style on
`tests\conftest.py`.
