---
name: add-model
description: "Add a new AI model or capability to the LocalAIExecution local-AI platform, following its core/adapter architecture. Use whenever the user wants to add, register, support, or wire up a new local model — another text-to-image model beside FLUX schnell/dev, a model variant, or a new modality/capability (text-to-speech, upscaling, etc.). Triggers on 'add model X', 'support SDXL', 'add a new capability', 'register a new model', 'plug in another local model', 'wire up a new modality'. Not for changing generation flags, fixing GPU/CUDA setup, or non-LocalAIExecution repos."
---

# Add a Model / Capability to LocalAIExecution

LocalAIExecution is a **reusable core** (`src\localai\core`) plus pluggable
**capability adapters** grouped by **modality** under
`src\localai\capabilities\<modality>\<capability>\` (e.g. `image\text_to_image\`).
The whole point of that seam is that adding a model is **additive**: a new
`ModelSpec` (and sometimes a new capability package + one manifest import line) —
with **no edits to the core**. This skill walks that path so the result matches
the conventions the existing FLUX text-to-image capability already follows.

The single most reliable reference is the **live code**, not this document. The
text-to-image capability under `src\localai\capabilities\image\text_to_image\` is
a complete worked example — read it when you write code. This skill is
**self-contained**: everything you need to do the task is here and in
`references\templates.md`. The capability layout is grouped by modality and can
grow new groups over time, so **list `src\localai\capabilities\` yourself** to see
the current groups rather than trusting any hard-coded path here.

## Scope discipline (read before you build)

The platform ships **one capability today: text-to-image**. Other modalities
(img2img, inpainting, upscaling, video, audio, language, LoRA, GUI) must **not**
be built without an explicit ask — the seams exist, but the scope does not. So:

- If the user explicitly asked for a specific model/modality, that ask is your
  green light — proceed.
- If the request is vague ("add more models"), stop and ask **which** model/repo
  and **what modality**, plus whether it is gated on Hugging Face. Don't invent
  scope.

## Step 0 — Read the ground truth, then classify the request

First, read these so your changes match reality (the code is the source of
truth and may have moved on from any snippet here):

- `src\localai\core\interfaces.py` — the `CapabilityAdapter` protocol, `Artifact`,
  `InferenceRequest`.
- `src\localai\core\registry.py` — `ModelSpec` fields, `register_capability`,
  `get_model`, `default_model`.
- The whole `src\localai\capabilities\image\text_to_image\` package — your template.
- `src\localai\capabilities\__init__.py` and `src\localai\capabilities\image\__init__.py`
  — the modality-grouped registration manifests you'll mirror.

Also **list the `src\localai\capabilities\` tree** so you can see the current
modality groups (today `image\`) and place your files correctly.

Then decide which of the two cases you're in. This determines the entire
workflow, so get it right:

| | **Case A — new model in an existing capability** | **Case B — new capability / modality** |
|---|---|---|
| Trigger | Same modality as an existing adapter (another text-to-image model) and the same pipeline behaviour fits the existing adapter | A different modality, OR a model whose pipeline/IO doesn't fit any existing adapter |
| Work | Add a `ModelSpec`; branch the existing adapter only where the new model genuinely differs | New package `src\localai\capabilities\<modality>\<name>\` + one manifest import line |
| Core edits | none | none |
| Example | adding `dev` beside `schnell` in `image\text_to_image\models.py` | adding an `audio\text_to_speech\` package under a new `audio\` modality |

When unsure, ask yourself: *does the existing adapter's `load_pipeline` + `run`
already do the right thing for this model, given a new `ModelSpec`?* If yes →
Case A. If it would need a fundamentally different pipeline class, request shape,
or artifact type → Case B.

The detailed, copy-shaped skeletons for both cases live in
`references\templates.md`. Read that file when you start writing code.

## The contract you implement (both cases share it)

Every capability satisfies the structural protocol in `core\interfaces.py`:

```python
class CapabilityAdapter(Protocol):
    capability_id: str          # e.g. "text-to-image"
    display_name: str
    def list_models(self) -> list[ModelSpec]: ...
    def register_cli(self, subparsers, shared_parents) -> None: ...
    def build_request(self, model_spec, settings) -> InferenceRequest: ...
    def load_pipeline(self, model_spec, device, dtype, offload): ...
    def run(self, pipeline, request) -> tuple[list[Artifact], ProvenanceRecord]: ...
```

You get all of this **for free** from the core and must reuse it rather than
reinventing it:

- **Registry** (`core\registry.py`) — `ModelSpec`, `register_capability(adapter)`
  (call it at import time), `get_model`, `default_model`.
- **Config** (`core\config.py`) — `load_settings(...)` resolves the layered
  precedence (CLI > env `LOCALAI_*` > file > builtin) into a `Settings` object
  with typed accessors `get_int` / `get_float` / `get_str`.
- **Engine** (`core\engine.py`) — `Engine().load(...)` / `.run(request)` /
  `.unload(...)`. The engine owns device/dtype/offload and fills the shared
  provenance fields (device, dtype, offload, load time, library versions).
- **Output** (`core\output.py`) — `write_artifact(artifact, record, settings)`
  builds a collision-safe path and writes the `.json` sidecar. For a brand-new
  payload type, register a writer with `register_writer("<type>", fn, "<ext>")`.
- **Errors** (`core\errors.py`) — raise the typed errors so failures get the
  right exit code (see the table below).
- **CLI** (`core\cli.py`) — `emit_result(json_mode, cap_id, model_id, written)`
  prints the shared one-shot / `--json` contract. `register_cli` receives
  `shared_parents = [global_parent, common_gen_parent]`, which already provide
  `--json`, `--config`, `-v`, `--model`, `--output-dir`, `--seed`, `--dtype`,
  `--offload`, `--batch`. Add `parents=shared_parents` to your subparser so your
  command inherits them; only add flags that are specific to your modality.

## Case A — add a model variant to an existing capability

This is how `dev` was added beside `schnell`. Usually a few lines, no new files.

1. **Add a `ModelSpec`** to the capability's `models.py` (e.g.
   `capabilities\image\text_to_image\models.py`) and include it in the
   exported `MODELS` list. Set the fields that describe how this model differs:
   `repo`, `pipeline_class`, `gated`, step/guidance/negative-prompt support,
   `recommended_dtype`, size constraints, and `is_default` (only one default per
   capability). Put genuinely model-specific guidance in `notes`.
2. **Branch the adapter only where the model truly differs.** Look at how the
   FLUX adapter keys behaviour off the spec (`supports_guidance`,
   `supports_negative_prompt`, `max_sequence_length`) rather than hard-coding per
   model. Prefer adding a spec flag and reading it over `if model_id == "...":`.
   If the new model needs a different pipeline class, make `load_pipeline` import
   and construct it based on `model_spec.pipeline_class`.
3. **Tests** — extend the capability's `test_models.py` (or equivalent) to assert
   the new model resolves and carries the right flags. Keep it GPU-free.

No import-line change, no new writer (you emit the same artifact type), and the
default model only changes if you deliberately move `is_default`.

## Case B — add a new capability

Capabilities live under a **modality group**:
`src\localai\capabilities\<modality>\<capability>\`. Before you write anything,
decide whether your capability belongs to an **existing modality** (today only
`image\`) or needs a **new modality** folder (e.g. `audio\`, `video\`). That
choice changes only the one registration line in step 2.

1. **Create the capability package** at
   `src\localai\capabilities\<modality>\<your_capability>\` with:
   - `models.py` — `CAPABILITY_ID` and one `ModelSpec` per model, exported as
     `MODELS`.
   - `adapter.py` — your `CapabilityAdapter` class implementing the five methods,
     a typed `InferenceRequest` subclass for your inputs, and helper functions.
     Call `register_capability(YourAdapter())` at module bottom (import side
     effect). Import sibling modules by full path, e.g.
     `from localai.capabilities.<modality>.<your_capability> import writer`.
   - `cli.py` — a `register(adapter, subparsers, shared_parents)` that adds your
     subcommand(s), maps args → `load_settings` overrides, drives `Engine`, and
     writes results with `write_artifact` + `emit_result`.
   - `writer.py` *(only if you emit a new artifact type)* — a `write_fn(artifact,
     path, record)` registered via `register_writer("<type>", write_fn, "<ext>")`.
     The core still writes the `.json` sidecar for you. (Reuse the existing
     `image` writer instead if you produce PNGs.)
   - Optional extras the FLUX package keeps separate for clarity: a `sizes.py`
     equivalent for input validation, a `repl.py` for an interactive mode.
2. **Register it — the one line.** Importing a capability module self-registers
   its adapter, so you only add an import to the right **manifest**:
   - **Existing modality** (e.g. another capability under `image\`): add to the
     modality manifest `src\localai\capabilities\<modality>\__init__.py`:
     ```python
     from localai.capabilities.<modality> import your_capability  # noqa: F401
     ```
     and add the name to that file's `__all__`.
   - **New modality** (e.g. `audio\`): also create the modality manifest
     `src\localai\capabilities\<modality>\__init__.py` that imports your capability
     (as above), then add one line to the top-level
     `src\localai\capabilities\__init__.py`:
     ```python
     from localai.capabilities import your_modality  # noqa: F401
     ```
     and add it to that file's `__all__`.

   Either way the dispatcher auto-discovers it; `localai capabilities` lists it
   and your subcommands appear under `localai`. The core is never touched.
3. **Tests** — add GPU-free tests (see Testing) covering protocol conformance,
   model resolution, and any pure logic (request building, input validation).

## Rules of thumb (this is where adapters go wrong)

These mirror the FLUX implementation; they matter because the platform is
GPU-first, scriptable, and must stay fast to import.

- **Keep heavy imports lazy.** Import `torch`, `diffusers`, model libraries
  **inside** `load_pipeline` / `run`, never at module top. Registry/CLI discovery
  must stay fast and GPU-free, or you break `localai capabilities` and the whole
  test suite on machines without CUDA.
- **Verify CUDA before loading.** Call `core.gpu.verify_cuda()` at the top of
  `load_pipeline` so there is no silent CPU fallback — that's a deliberate
  platform guarantee.
- **Raise typed errors.** Wrap upstream load/runtime failures and map them to
  `core.errors` types (gated → `GatedModelError`, download → `NetworkError`,
  CUDA OOM is auto-mapped by the engine, bad input → `InvalidArgumentError`).
  Copy the FLUX adapter's `_map_load_error` classification approach for gated
  models so a missing HF token surfaces as exit code 6 with an actionable remedy.
- **Set sane VRAM defaults per capability.** If your model's footprint is large,
  set `capability_defaults = {"offload": "model"}` on the adapter (FLUX does this
  because its ~33 GB bf16 footprint oversubscribes a 32 GB card). Don't force
  `offload=none`.
- **Return a `ProvenanceRecord` with your `params`.** Fill the capability-specific
  block (prompt, steps, voice, duration, …) and the fields you own (seed,
  timestamp, `generate_seconds`, `model_repo`). The engine fills device, dtype,
  offload, load time, and library versions — don't duplicate those.
- **Don't touch the cross-capability surface.** `--json`, the one-shot
  "final stdout line is the saved path" contract, output-path construction, and
  the exit-code map are stable because a marketplace skill depends on them. Put
  all model-specific knobs inside your capability.
- **Prefer `settings.get_int/get_float/get_str` in `build_request`.** They coerce
  for you, so you usually don't need to touch `core\config.py`. Only if you want a
  brand-new numeric knob to be settable via env (`LOCALAI_<KEY>`) or the config
  file with auto-coercion do you add it to `KEY_TYPES` in `core\config.py` — that
  is the one tiny, optional config-plumbing exception to "no core edits".

## Testing (must stay GPU-free)

The suite runs without a GPU and must keep doing so — that's how it stays a fast
(~4 s) gate. Model that on `tests\conftest.py`, which defines a dummy adapter
that exercises the core with no torch/CUDA.

- **Protocol conformance:** assert `isinstance(YourAdapter(), CapabilityAdapter)`
  (see `tests\test_interfaces.py`).
- **Registration & resolution:** after `registry.discover_capabilities()`, assert
  `get_model(cap_id, model_id)` returns your spec with the right flags, your
  capability appears in `list_capabilities()`, and `default_model(cap_id)` is what
  you expect (mirror `tests\test_models.py`).
- **Pure logic:** unit-test request building and any input validation directly,
  without loading a pipeline.
- **Never import torch/diffusers at test-collection time.** If a test needs the
  pipeline, it belongs in a GPU/integration lane, not the default suite.

Run the targeted tests first, then the full GPU-free suite:

```powershell
.venv\Scripts\python.exe -m pytest tests\test_<your_capability>.py
.venv\Scripts\python.exe -m pytest
```

## Verify and finish

1. `localai capabilities` (or `localai capabilities --json`) — your capability and
   model(s) are listed.
2. `.venv\Scripts\python.exe -m pytest` — all GPU-free tests pass.
3. For Case B, sanity-check the help wiring: `localai <your-subcommand> --help`.
4. If a real GPU is available and the user wants it, do one live generation to
   confirm the end-to-end path and the saved-path/`--json` contract. Outputs land
   in git-ignored `outputs\`; **never commit `.venv\`, `outputs\`, the HF cache,
   or any token.**
5. Tell the user exactly what changed (files added/edited, the one import line for
   Case B), how to invoke the new model, whether it is gated (needs an HF token),
   and the test result.

## Exit codes (for error mapping and for telling the user what to expect)

`0` ok · `1` unexpected · `2` invalid args · `3` CUDA/torch wrong build ·
`4` GPU absent · `5` OOM · `6` gated/token · `7` network/download ·
`8` unknown capability/model.

## Pointers

- `references\templates.md` — copy-shaped skeletons for Case A (a `ModelSpec` +
  the adapter branch points) and Case B (the modality manifest, `models.py`,
  `adapter.py`, `cli.py`, `writer.py`, tests).
- `src\localai\capabilities\image\text_to_image\` — the real, working reference;
  `src\localai\capabilities\__init__.py` and `image\__init__.py` show the
  modality-grouped manifests.
