# Local AI Execution Platform — Plan (Wide-Scope)

> Build a dedicated local-AI platform in its own repo at `c:\Dev\MyRepos\LocalAIExecution` — a reusable, modality-agnostic core/runtime plus pluggable model **adapters** — on a Blackwell RTX 5090, and ship **text-to-image (FLUX) as its first and only adapter** so that future local models plug in as a new adapter module + one registry entry without touching the core.

**Created**: 2026-06-28
**Approach**: WideScope-RefactorImprovements (greenfield — interpreted as a clean platform/adapter architecture and the broadest spec-aligned feature set for the first capability, building the seams future models/skills plug into without exceeding the spec's Non-Goals)
**Spec**: ./LocalAIExecution-spec.md
**Review Score**: 9/10 — PASS (2026-06-28)

## Architecture plan

A `src/`-layout package `localai` is split into a reusable, modality-agnostic **core/runtime** (`localai.core`) and pluggable, self-contained **capability adapters** (`localai.capabilities.*`). The core owns everything shared across models: the GPU/CUDA bootstrap and verification (`gpu`), the capability/model **registry** keyed by capability id + model id (`registry`), layered configuration with global → per-capability → per-model precedence (`config`), the generic artifact + provenance/metadata record with its sidecar-JSON writer and collision-safe filenames (`metadata`/`output`), typed errors with deterministic exit codes (`errors`), the resident **engine** that loads a pipeline once and reuses it (`engine`), and the top-level CLI dispatcher with the shared `--json` contract (`cli`). Each capability is a self-contained module that registers itself with the core and implements one small interface, `CapabilityAdapter`: declare its id + models/params, contribute its own CLI subcommands, build a typed request, `load_pipeline(...)`, and `run(...)` returning an `Artifact` + provenance record. **Text-to-image is the first and only adapter built now** — it holds all FLUX-specific behavior (schnell default vs. dev opt-in, guidance/negative-prompt rules, steps/size, the PNG image writer) under `localai.capabilities.text_to_image`. Adding a future model — another image model today, or a non-image modality later — is a **new adapter module plus one registration line**, with no edits to the core or to existing adapters. The image PNG + sidecar-JSON writer is the only concrete artifact writer implemented now; the artifact/metadata seam is designed so other output types can be described later, but is kept practical, not speculative. The hardest platform risk — a Blackwell sm_120 (RTX 5090) CUDA stack — is retired first via a GPU-aware PowerShell bootstrap that installs the cu128 PyTorch wheel and gates on a real CUDA-availability verification before any model work. Scope stays text-to-image only; audio/language/img2img/inpainting/upscaling/video/LoRA/GUI are explicitly **not** built — only the seams they would later plug into.

## [x] Phase 1: Platform scaffold & packaging skeleton
> Stand up the standalone repo with the core/capabilities split that installs as a single console command — before any GPU or model code.

**Milestone**: `c:\Dev\MyRepos\LocalAIExecution` exists as a git repo, installs cleanly in editable mode (minus torch) into a fresh venv, and exposes a `localai` command that runs and prints help.
**Acceptance**:
- The package installs in editable mode into a fresh venv and the `localai` entry point resolves.
- The package and each of its subpackages (`localai.core`, `localai.capabilities`) import cleanly (stubs allowed).
- Repo has `.gitignore`, README stub, license, and an example layered config; venv/outputs/model-cache are ignored.

### [x] Step 1.1: Create repo with core + capabilities source layout
- **What**: Create the project root and a `src/localai/` package split into an importable `core/` (the runtime seams) and a `capabilities/` package (the adapter slots), plus `tests/`, `scripts/`, and `docs/`. Initialize git.
- **Deliverables**: `c:\Dev\MyRepos\LocalAIExecution\src\localai\__init__.py` (holds `__version__`); `src\localai\core\__init__.py` and core stubs `cli.py`, `config.py`, `registry.py`, `interfaces.py`, `engine.py`, `output.py`, `metadata.py`, `errors.py`, `gpu.py`; `src\localai\capabilities\__init__.py` (the registration manifest); `tests\`, `scripts\`, `docs\`; an initialized git repository.
- **Dependencies**: None
- **Verify**: after install, the package and each of its subpackages import cleanly; the repository is a clean, initialized git repo with venv, outputs, and the model cache git-ignored.

### [x] Step 1.2: Author pyproject with console-script entry point
- **What**: Define packaging metadata, runtime dependencies (excluding torch, which the bootstrap installs from the cu128 index), and a console-script entry point mapping `localai` to `localai.core.cli:main`. Pin tested minimum versions for `diffusers`, `transformers`, `accelerate`, `safetensors`, `huggingface_hub`, `pillow`, `sentencepiece`, `protobuf`.
- **Deliverables**: `c:\Dev\MyRepos\LocalAIExecution\pyproject.toml` (`[project]`, `[project.scripts] localai = "localai.core.cli:main"`, `[tool.setuptools]` src-layout); minimal `core\cli.main` stub that prints help and exits 0.
- **Dependencies**: 1.1
- **Verify**: after the package is installed in editable mode, the `localai` entry point resolves and printing its help exits 0 (success).

### [x] Step 1.3: Repo hygiene files
- **What**: Add `.gitignore` (ignore `.venv/`, `outputs/`, `__pycache__/`, the model cache, output `*.png`/`*.json` sidecars), a license note for the tool, and a README stub naming the project as a local-AI platform with text-to-image as its first capability.
- **Deliverables**: `c:\Dev\MyRepos\LocalAIExecution\.gitignore`, `LICENSE` (default MIT — see Open Questions), `README.md` (stub).
- **Dependencies**: 1.1
- **Verify**: venv and outputs are git-ignored (not surfaced as tracked files); the README renders.

### [x] Step 1.4: Example layered configuration file
- **What**: Add a commented example TOML showing the layered shape — a global `[defaults]` table, a per-capability table (e.g. `[text-to-image]`), and per-model tables (e.g. `[text-to-image.models.schnell]`) — capturing every default key (model, size/preset, steps, dtype, offload, output dir, seed policy). No loading logic yet (Phase 3 consumes it).
- **Deliverables**: `c:\Dev\MyRepos\LocalAIExecution\config.example.toml`; documented section/key names that `core\config.py` will later read.
- **Dependencies**: 1.1
- **Verify**: the example config parses as valid TOML.

## [x] Phase 2: GPU-aware bootstrap & CUDA stack
> Solve the hardest platform risk first: install a PyTorch build that actually drives the Blackwell sm_120 GPU, and gate on a real CUDA verification — before any registry/engine/adapter work.

**Milestone**: A one-command bootstrap creates the venv, installs the correct cu128 torch, installs the package, and proves CUDA is available on the RTX 5090.
**Acceptance**:
- Running the bootstrap from a clean state ends with CUDA verified True and the reported device name containing "RTX 5090".
- The installed torch reports CUDA 12.8+ and lists sm_120 in its arch list (no CPU fallback).
- A tiny CUDA tensor smoke runs on-device, confirming the runtime stack works (full-model warm is added in Phase 5).

### [x] Step 2.1: GPU detection & CUDA verification module + `doctor`
- **What**: Implement a core module that probes the NVIDIA GPU (name, driver, compute capability) and, post-install, verifies torch sees CUDA, the device is the RTX 5090, and capability `(12, 0)` / sm_120 is in torch's supported arch list. On failure it raises a clear error that the typed-error layer (Phase 3) later maps to exit code 3 (CUDA/torch wrong build) or 4 (GPU absent). In this phase `doctor` is wired into a minimal stub parser; Step 4.3 folds it into the full CLI dispatcher, so there is no rework surprise.
- **Deliverables**: `src\localai\core\gpu.py` functions `detect_nvidia_gpu()` (parses `nvidia-smi`) and `verify_cuda()` (checks `torch.cuda.is_available()`, device name, capability vs `torch.cuda.get_arch_list()`); a core `doctor` subcommand running both.
- **Dependencies**: 1.2
- **Verify**: the GPU doctor check reports the RTX 5090 with CUDA available and the cu128 stack verified — on this machine the observed driver 591.86 and compute capability 12.0 are expected values for this hardware, not hard requirements.

### [x] Step 2.2: Bootstrap script with cu128 torch install
- **What**: Author a PowerShell bootstrap that checks for Python 3.12 (3.12.10 observed on this machine), creates `.venv`, detects the GPU, installs torch from the stable cu128 index `https://download.pytorch.org/whl/cu128`, and on verification failure retries from the nightly cu128 index `https://download.pytorch.org/whl/nightly/cu128`. Deliberately avoids xformers (rely on PyTorch SDPA) to dodge sm_120 incompatibilities.
- **Deliverables**: `c:\Dev\MyRepos\LocalAIExecution\scripts\bootstrap.ps1` (params: `-Nightly`, `-SkipSmoke`, `-Model`); the script installs torch from the cu128 index, then installs the package in editable mode.
- **Dependencies**: 2.1
- **Verify**: after a clean run, torch reports CUDA 12.8+ and CUDA is available on the RTX 5090.

### [x] Step 2.3: CUDA tensor smoke in bootstrap
- **What**: Extend the bootstrap to run the GPU doctor check and a tiny on-device tensor op (e.g. a small matmul moved to cuda) to prove the cu128/sm_120 stack executes on the GPU, failing loudly (non-zero exit) if CUDA is unavailable. Full default-model warm-up + real generation smoke is wired once the adapter exists (Phase 5).
- **Deliverables**: bootstrap runs the GPU doctor check then a minimal CUDA smoke; clear success/failure summary printed.
- **Dependencies**: 2.2
- **Verify**: Bootstrap finishes with a printed success summary; forcing CPU-only (hiding the GPU) makes it exit non-zero with a clear message.

## [x] Phase 3: Core platform seams — adapter interface, registry, config, errors
> Establish the platform/adapter contract and the pure-Python core seams — no GPU — so the engine and the text-to-image adapter are built *on top of* a stable, tested core.

**Milestone**: The `CapabilityAdapter` interface, the capability/model registry keyed by capability + model, layered configuration precedence, and typed errors all exist and are covered by fast unit tests.
**Acceptance**:
- Unit tests pass for config precedence, registry lookups by `(capability, model)`, and adapter-interface conformance against a dummy in-test adapter.
- Registering a hypothetical new capability/model is a single manifest entry + adapter module with no change to core modules (demonstrated by a test).
- Every failure mode maps to a documented, deterministic exit code.

### [x] Step 3.1: Capability/adapter interface (the common contract)
- **What**: Define the small interface every capability implements: declare `capability_id` and `display_name`, `list_models()` (its model specs), `register_cli(subparsers, shared_parents)` (contribute subcommands), `build_request(model_spec, settings)` (→ typed request), `load_pipeline(model_spec, device, dtype, offload)` (→ loaded pipeline), and `run(pipeline, request)` (→ `Artifact` + provenance record). Also define the `Artifact` (type tag + in-memory payload) and base request/record types here.
- **Deliverables**: `src\localai\core\interfaces.py`: `CapabilityAdapter` protocol/ABC, `Artifact` dataclass, base `InferenceRequest` and `ProvenanceRecord` references; a dummy adapter under `tests\` for conformance checks.
- **Dependencies**: 1.1
- **Verify**: A unit test instantiates the dummy adapter and asserts it satisfies `CapabilityAdapter` (all required members present and callable).

### [x] Step 3.2: Capability/model registry keyed by capability + model
- **What**: Implement a registry holding registered capabilities and their models, keyed by capability id + model id. Capabilities self-register on import via the `capabilities` manifest; lookups raise a typed error for unknown capability or model. A `ModelSpec` captures per-model behavior (short name, HF repo, pipeline class, default/allowed steps, guidance support+default, negative-prompt support, gated flag, default size, max sequence length, recommended dtype).
- **Deliverables**: `src\localai\core\registry.py`: `ModelSpec` dataclass, `register_capability(adapter)`, `get_capability(id)`, `list_capabilities()`, `get_model(capability_id, model_id)`, `list_models(capability_id)`, `discover_capabilities()` (imports `src\localai\capabilities\__init__.py`).
- **Dependencies**: 3.1
- **Verify**: Unit tests confirm a dummy capability + model register and resolve by `(capability, model)`, and that unknown ids raise the registry error.

### [x] Step 3.3: Layered configuration with capability/model precedence
- **What**: Implement a config loader producing an effective settings object with precedence (decreasing): CLI args > env vars (`LOCALAI_*`) > config file (per-model table > per-capability table > global `[defaults]`) > built-in defaults (model-spec default > capability default > core default). Validate types and ranges.
- **Deliverables**: `src\localai\core\config.py`: `Settings` object, `load_settings(capability_id, model_id, cli_overrides, config_path)`, precedence resolver, env-var mapping.
- **Dependencies**: 1.4
- **Verify**: Unit test asserts CLI beats env beats per-model beats per-capability beats global beats default for the same key.

### [x] Step 3.4: Typed errors & deterministic exit codes
- **What**: Define an exception hierarchy with stable exit codes and actionable messages for: invalid args (2), CUDA/torch unavailable or wrong build (3), GPU not detected (4), out-of-memory (5), gated/token denied (6), network/download failure (7), unknown/invalid capability or model (8); 0 success, 1 unexpected. The CLI top-level catches these and prints the message (no raw traceback) while returning the code.
- **Deliverables**: `src\localai\core\errors.py`: base `LocalAIError` with `exit_code`, concrete subclasses, and a `handle_error()` used by `core.cli.main`.
- **Dependencies**: 1.2
- **Verify**: Unit tests assert each error type carries its documented exit code and a non-empty, actionable message.

## [x] Phase 4: Core artifact/provenance, resident engine & CLI dispatch
> Build the modality-agnostic runtime — provenance/output, the load-once engine, the top-level dispatcher, and the shared `--json` contract — and prove it adapter-agnostic with a fake adapter (no GPU).

**Milestone**: A dummy in-test capability drives the full path — dispatch → resident engine → artifact write → `--json` — with no GPU, proving the core carries no image-specific assumptions.
**Acceptance**:
- A dummy capability registers a subcommand, runs through the engine (one load, reused across calls), and writes a fake artifact + sidecar — all without GPU.
- `--json` emits a single uniform JSON object (capability, model, artifact paths + provenance) on stdout and nothing else on stdout.
- Unknown capability/model and a missing writer each surface a typed error with the documented exit code.

### [x] Step 4.1: Generic artifact, provenance & writer-selection seam
- **What**: Define a modality-agnostic provenance record (capability id, model id + repo, seed, ISO timestamp, load/generate durations, device/dtype/offload, torch/diffusers/transformers + tool versions) with a nested capability-specific `params` block, plus an artifact-writer selection map keyed by artifact type. Implement the generic sidecar-JSON writer and collision-safe filenames; the only concrete writer (image) is registered later by the text-to-image capability (Step 5.3).
- **Deliverables**: `src\localai\core\metadata.py`: `ProvenanceRecord` dataclass + `to_json()`; `src\localai\core\output.py`: `register_writer(artifact_type, writer)`, `write_artifact(artifact, record, settings)` (selects writer + writes sidecar), `build_filename(...)` (timestamp + capability + model + seed + counter + extension).
- **Dependencies**: 3.1, 3.3, 3.4
- **Verify**: Unit tests cover filename uniqueness across same-second/batch calls and JSON round-trip; writing an unknown artifact type raises a typed error.

### [x] Step 4.2: Core resident engine (load-once lifecycle)
- **What**: Implement the engine that selects device (cuda) and dtype (bf16 default), and owns a pipeline cache keyed by `(capability_id, model_id)`: `load(...)` delegates pipeline construction to the owning adapter's `load_pipeline(...)` and caches it; `run(request)` routes to the adapter's `run(...)`; `unload(...)` frees CUDA memory for VRAM hygiene. Adapter load failures (network, gated, OOM) are wrapped in the Phase 3 typed errors.
- **Deliverables**: `src\localai\core\engine.py`: `Engine` class, `load(capability_id, model_id)`, `run(request)`, `unload(...)`, internal cache, device/dtype/offload setup.
- **Dependencies**: 3.2, 3.4, 2.1
- **Verify**: A unit test drives the engine through a dummy adapter (no GPU): two `run` calls share one cached pipeline; `unload` evicts it.

### [x] Step 4.3: Top-level CLI dispatcher (core + capability subcommands)
- **What**: Implement the dispatcher that builds the top-level parser, registers core subcommands (`doctor` — folded in here from the Phase 2 minimal stub parser — and `capabilities`), calls `discover_capabilities()`, then asks each registered adapter to contribute its own subcommands via `register_cli(...)` — so new capabilities add commands without touching the dispatcher. Top-level errors are caught and rendered via `handle_error`.
- **Deliverables**: `src\localai\core\cli.py`: `main()`, parser assembly, capability subcommand wiring, `capabilities` listing command.
- **Dependencies**: 3.2, 3.4
- **Verify**: with a dummy capability registered, the top-level help lists its subcommand and the capabilities listing shows the capability and its models; an unknown subcommand exits 2.

### [x] Step 4.4: Shared `--json` result contract
- **What**: Add a global `--json` flag (shared by all capabilities) that suppresses human chatter on stdout and emits one JSON object — capability, model, and an `artifacts` array of `{path, type, metadata}` — for unattended/skill use; diagnostics go to stderr. Define this as the stable cross-capability contract.
- **Deliverables**: `src\localai\core\cli.py` JSON-rendering branch reading the engine result + provenance; documented JSON schema.
- **Dependencies**: 4.1, 4.3
- **Verify**: with the dummy capability, the machine-readable `--json` mode emits a single JSON object containing artifact path(s) and provenance on stdout, with no non-JSON text.

## [x] Phase 5: Text-to-image adapter (first capability) — FLUX one-shot
> Build the first concrete adapter on top of the core — FLUX text-to-image — delivering the first real end-to-end GPU capability: prompt → saved PNG → printed absolute path (plus `--json`).

**Milestone**: The one-shot `generate` path loads schnell via the core engine, generates on the GPU, saves a PNG plus its sidecar through the image writer, and prints the saved absolute path as the final stdout line (or a JSON record in `--json` mode).
**Acceptance**:
- A one-shot run on this GPU produces a valid PNG plus sidecar and prints the absolute output path.
- Knobs (model, steps, size/preset, seed, output dir, batch count) take effect, and schnell correctly runs with guidance 0 / no negative prompt.
- Same prompt + seed + settings reproduces an identical image (assuming the same library/model versions, hardware, and dtype — bf16 has minor nondeterminism); bootstrap now warms schnell and runs a real generation smoke.

### [x] Step 5.1: Register the text-to-image capability + FLUX model specs
- **What**: Create the self-contained `text_to_image` capability that registers itself via the manifest and declares its models as registry entries under capability id `text-to-image`: `schnell` (`black-forest-labs/FLUX.1-schnell`, bf16, ~4 steps, guidance 0, no negative prompt, ungated/Apache-2.0, default) and `dev` (`black-forest-labs/FLUX.1-dev`, gated, guidance ~3.5, ~20–50 steps, default ~28).
- **Deliverables**: `src\localai\capabilities\text_to_image\__init__.py` (registers the adapter), `adapter.py` (`TextToImageAdapter` skeleton + `capability_id`/`list_models`), `models.py` (the two `ModelSpec` entries); one import line added to `src\localai\capabilities\__init__.py`.
- **Dependencies**: 3.1, 3.2
- **Verify**: the capabilities listing shows `text-to-image` with models `schnell` (default) and `dev`; a unit test resolves both via the registry's `get_model` lookup for `(text-to-image, schnell)` and `(text-to-image, dev)`.

### [x] Step 5.2: Size presets & dimension validation (capability-local)
- **What**: Implement aspect presets (square, widescreen, portrait) mapping to concrete width/height, plus explicit `--width/--height` overrides validated to FLUX-appropriate multiples. Invalid sizes raise the typed argument error (exit 2). Kept inside the capability since sizing is image-specific.
- **Deliverables**: `src\localai\capabilities\text_to_image\sizes.py`: preset table, `resolve_size(preset, width, height, model_spec)`.
- **Dependencies**: 5.1
- **Verify**: Unit test maps each preset to expected dimensions and rejects non-multiple sizes.

### [x] Step 5.3: Image artifact writer (PNG + embedded params + sidecar)
- **What**: Implement the concrete image writer for artifact type `image` and register it into the core writer-selection map: encode the PIL image to PNG, embed the provenance params as PNG text chunks, and rely on the core sidecar-JSON writer for the `.json` companion. This is the only concrete writer built now.
- **Deliverables**: `src\localai\capabilities\text_to_image\writer.py`: image writer registered for type `image`; prompt-derived slug sanitization.
- **Dependencies**: 4.1, 5.1
- **Verify**: Unit test writes a small image, re-reads its embedded PNG params, and confirms the sidecar JSON round-trips; slug sanitization strips unsafe characters.

### [x] Step 5.4: Adapter `load_pipeline` + model-aware `run`
- **What**: Implement `load_pipeline` (build the diffusers FLUX pipeline on cuda/bf16 with the configured offload, default none — fits 32 GB) and `run` (build a seeded `torch.Generator`; assemble only the kwargs the model supports — schnell: steps + `max_sequence_length`, guidance 0, no negative prompt; dev: steps + guidance + optional negative prompt; honor batch count; time the run; return `Artifact(image)` + `ProvenanceRecord`).
- **Deliverables**: `src\localai\capabilities\text_to_image\adapter.py`: `load_pipeline(...)`, `run(...)`, `build_request(...)`; seed handling (explicit or drawn-and-recorded).
- **Dependencies**: 4.2, 5.1, 5.2
- **Verify**: Loading schnell places the pipeline on cuda/bf16 once; same prompt+seed+settings reproduces an identical image; a schnell run completes in roughly a few seconds post-load.

### [x] Step 5.5: `generate` subcommand + bootstrap model-warm/smoke
- **What**: Contribute the `generate` subcommand via the adapter's `register_cli` (parse prompt + all knobs, resolve effective settings, run the engine, write outputs, print each saved absolute path — `--json` emits the shared record). Extend `scripts\bootstrap.ps1` to pre-download/warm the default schnell model into the HF cache and run a real `--smoke` generation; gated dev is not auto-downloaded.
- **Deliverables**: `src\localai\capabilities\text_to_image\cli.py`: `generate` subparser + arg-to-settings mapping; `scripts\bootstrap.ps1` warm + `--smoke` step writing under `c:\Dev\MyRepos\LocalAIExecution\outputs\`.
- **Dependencies**: 4.3, 4.4, 5.3, 5.4, 2.3
- **Verify**: the one-shot `generate` path writes a PNG plus sidecar and prints its saved absolute path as the final stdout line; an invalid `--steps` value exits 2; re-running the bootstrap offline still generates from the cache (cache hit).

## [x] Phase 6: Interactive mode & runtime robustness
> Add resident interactive reuse and harden the runtime's failure modes — gated dev access, OOM/offload, precision/performance, full error UX — into actionable messages with deterministic exit codes.

**Milestone**: The interactive session loads the model once and generates an image per prompt with no reload; the gated dev model is switchable with a token; and OOM/gated/network/CUDA failures produce clear, actionable messages, not raw tracebacks.
**Acceptance**:
- The model loads exactly once in interactive mode; in-loop overrides and model switching (schnell↔dev) apply without restarting, and a clean exit leaves no orphaned VRAM.
- With a valid HF token + accepted license, selecting the dev model via `--model dev` generates higher-fidelity images at higher step counts; without access it exits 6 with remediation steps.
- A forced OOM triggers the documented offload/size fallback (or exits 5 with a remedy), and every spec'd failure maps to its exit code.

### [x] Step 6.1: Interactive REPL scaffold (resident load-once)
- **What**: Add the capability's `interactive` subcommand: load the configured model once via the core engine, then read prompts in a loop, generating and saving each and printing the path. Handle EOF/`exit`/`quit` and keyboard interrupt gracefully.
- **Deliverables**: `src\localai\capabilities\text_to_image\repl.py`: `run_repl(settings, engine)`; `interactive` subparser in the capability CLI.
- **Dependencies**: 5.4, 5.5
- **Verify**: Two successive prompts produce two images with only one model-load logged.

### [x] Step 6.2: In-loop overrides & control commands
- **What**: Support per-prompt inline overrides and control commands (set steps/size/seed, show current settings, switch model) parsed from the input line, reusing the core config precedence and the capability's size resolver. Unknown commands print help, not a traceback.
- **Deliverables**: `src\localai\capabilities\text_to_image\repl.py` command parser; reuse of `core.config` + `text_to_image.sizes`.
- **Dependencies**: 6.1, 3.3, 5.2
- **Verify**: Changing steps mid-session changes the next image's recorded steps; bad input prints guidance and continues the loop.

### [x] Step 6.3: Model switching & VRAM hygiene
- **What**: Allow switching the active model in-session; on swap, call the core engine's `unload(...)` to release the previous pipeline and free CUDA memory before loading the next, keeping resident reuse for the active model.
- **Deliverables**: REPL `model` command wiring to `Engine.unload(...)` + `Engine.load(...)`.
- **Dependencies**: 6.1, 4.2
- **Verify**: Switching schnell↔dev reloads only on switch (not per prompt) and VRAM returns to baseline after unload.

### [x] Step 6.4: Gated dev model access & token UX
- **What**: Resolve the HF token from env (`HF_TOKEN`/`HUGGING_FACE_HUB_TOKEN`) or the huggingface CLI cache during `load_pipeline`; on gated denial or missing token, raise the gated error (exit 6) with steps to accept the license on the model page and supply a token. schnell stays fully ungated.
- **Deliverables**: token resolution in the adapter's `load_pipeline`; mapping of HF gated/auth exceptions to the typed gated error; README pointer.
- **Dependencies**: 5.4, 3.4
- **Verify**: Requesting dev without a token exits 6 with actionable text; with a valid token it loads and generates higher-fidelity images at higher steps.

### [x] Step 6.5: OOM, offload, precision & performance options
- **What**: Catch CUDA out-of-memory during load/generate and surface the typed OOM error (exit 5) naming concrete remedies (enable model/sequential CPU offload, VAE tiling, reduce size/batch), and honor an `offload = none|model|sequential` config to retry under tighter memory. Expose dtype (bf16 default) and attention backend (PyTorch SDPA, no xformers) as configurable knobs with documented speed/VRAM trade-offs.
- **Deliverables**: OOM handling + offload/VAE-tiling wiring in `core\engine.py` and the adapter; config keys for dtype/attention/offload; remedy text.
- **Dependencies**: 5.4, 3.3, 3.4
- **Verify**: With a forced tiny memory ceiling, generation either succeeds under sequential offload or exits 5 with the remedy — never a raw stack trace; changing dtype/offload changes recorded metadata and measurably changes VRAM/latency.

### [x] Step 6.6: Full error-UX pass
- **What**: Audit every entry point so each spec'd failure (CUDA wrong-build, GPU absent, OOM, gated/no-token, network/download, invalid size/steps, unknown capability/model) is caught and rendered as an actionable message with its deterministic exit code — never a raw traceback. Map upstream library exceptions (diffusers/transformers/huggingface_hub) to the typed errors.
- **Deliverables**: hardened `handle_error` coverage in `core\cli.py`; upstream-exception mapping table.
- **Dependencies**: 6.4, 6.5, 3.4
- **Verify**: A table-driven test confirms each simulated failure yields the right exit code and message.

## [x] Phase 7: Docs, tests & end-to-end validation
> Make the platform usable by a first-timer, callable by a future skill, and provably extensible — then prove the spec's success criteria end to end.

**Milestone**: A complete README, a green unit-test suite (including adapter-interface conformance and registry-by-capability+model), a documented skill-invocation contract, and an "adding a new capability" guide; a cold-start run produces an image and reproducibility holds.
**Acceptance**:
- A first-time user goes from nothing installed to a generated image by following the README, without hand-troubleshooting GPU libraries.
- All unit tests pass locally; the skill-invocation section documents the `--json` contract and every exit code; the guide shows a new capability is an adapter module + one manifest line.
- Cold-start one-shot, interactive resident-reuse, dev higher-fidelity, offline-after-cache, and same-seed reproducibility (assuming the same library/model versions, hardware, and dtype — bf16 has minor nondeterminism) are all demonstrated.

### [x] Step 7.1: Complete README & troubleshooting
- **What**: Write the full README, which explicitly owns the literal, copy-pasteable command examples (the cu128 torch install from `https://download.pytorch.org/whl/cu128`, and example `generate` / `interactive` / `--json` usage): platform overview (core + adapters; text-to-image as the first capability), requirements, bootstrap setup, usage, a model comparison table (schnell vs dev: steps, guidance, negative-prompt support, gated, speed/quality), configuration sections + precedence, output/naming/reproducibility, performance/VRAM tuning, and a troubleshooting matrix keyed to the exit codes.
- **Deliverables**: `c:\Dev\MyRepos\LocalAIExecution\README.md` (full); troubleshooting table; model table.
- **Dependencies**: 6.6
- **Verify**: A reviewer following only the README reaches a generated image on a comparable machine.

### [x] Step 7.2: Skill-invocation contract & adding-a-capability guide
- **What**: Document how a future skill invokes a capability — the exact machine-readable contract (a capability command invoked with the global `--json` flag), the stdout JSON schema, the full exit-code table, predictable non-colliding output paths, and offline/caching assumptions — without implementing any skill or plugin packaging. Add a short guide proving extensibility: implement `CapabilityAdapter`, register one line in the manifest, no core edits.
- **Deliverables**: `c:\Dev\MyRepos\LocalAIExecution\docs\skill-invocation.md` and `docs\adding-a-capability.md`; cross-links from README.
- **Dependencies**: 4.4, 3.4
- **Verify**: The documented JSON schema and exit codes match real CLI output (spot-checked); the guide's steps reference only adapter + manifest, not core modules.

### [x] Step 7.3: Finalize unit test suite
- **What**: Ensure the cheap, meaningful non-GPU unit tests are complete and green: config precedence, registry lookup by `(capability, model)`, filename/metadata round-trip, size validation, exit-code mapping, and adapter-interface conformance (the dummy adapter). No heavy GPU integration tests.
- **Deliverables**: `c:\Dev\MyRepos\LocalAIExecution\tests\` modules covering the above; a documented command in the README for running the non-GPU unit-test suite.
- **Dependencies**: 3.1, 3.2, 3.3, 3.4, 4.1, 5.2
- **Verify**: the non-GPU unit-test suite runs green locally and finishes quickly without requiring the GPU.

### [x] Step 7.4: End-to-end validation against success criteria
- **What**: Execute the spec's success criteria on the real GPU: cold-start one-shot prints a path; interactive loads once and reuses; dev at higher steps is visibly higher fidelity; same prompt+seed+settings reproduces an identical image (assuming the same library/model versions, hardware, and dtype — bf16 has minor nondeterminism); the tool runs offline after caching with no cloud calls.
- **Deliverables**: `c:\Dev\MyRepos\LocalAIExecution\docs\validation.md` checklist recording results (paths, timings, reproduced-hash match).
- **Dependencies**: 7.1, 6.3, 6.4
- **Verify**: Every success-criterion row is checked off with evidence (output paths, load-once log, matching seed reproduction).

## Risks & Mitigations
| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Stable cu128 wheel lacks sm_120 support at install time | GPU falls back to CPU; tool unusably slow | Medium | Bootstrap verifies CUDA + arch list and auto-retries from the nightly cu128 index; `doctor` gate blocks a broken stack |
| Capability seam over-generalized into speculative complexity | Wasted effort; harder-to-read core | Medium | Build exactly one adapter now; one concrete writer; validate the seam with a dummy in-test adapter, not extra production modules |
| FLUX bf16 VRAM pressure (esp. dev + batch) | Out-of-memory failures | Medium | Default no-offload fits 32 GB; configurable model/sequential CPU offload + VAE tiling + batch guard; typed OOM remedy |
| Large first-time downloads (schnell/dev + T5 text encoder) | Slow/failed first run | High | Warm step in bootstrap; offline-from-cache afterward; network errors mapped to exit 7 with guidance |
| dev model gating/token friction | dev path blocked | Medium | Token resolution + clear exit-6 remediation; dev kept optional, schnell ungated default |
| diffusers/transformers API drift breaks FLUX usage | Build breaks on upgrade | Low/Medium | Pin tested minimum versions in pyproject; record library versions in provenance; document tested set |
| xformers / third-party attention incompat with sm_120 | Install or runtime failures | Medium | Avoid xformers entirely; rely on built-in PyTorch SDPA attention |
| bf16 GPU nondeterminism undermines exact reproduction | Reproducibility weaker than promised | Low/Medium | Seed the generator and record full provenance; document that identical reproduction assumes same versions/hardware |

## Open Questions
- Default output directory: repo-local `outputs\` vs the user's Pictures folder — which is the better default for later skill consumption?
- Tool license: MIT vs Apache-2.0 (independent of model licenses — schnell is Apache-2.0, dev is a gated non-commercial license)?
- Should the bootstrap default to stable cu128 (with nightly fallback) or prefer nightly cu128 outright, given Blackwell's recency at execution time?
- Config file location precedence: prefer repo-local `localai.toml`, a user-home config path, or both with repo-local winning?
- Capability discovery: keep the explicit `capabilities\__init__.py` import manifest, or move to Python entry-points for third-party adapters once a second capability exists?
- Should the dev model be pre-warmed by the bootstrap (large + gated), or only downloaded on first explicit `--model dev` use (current plan: the latter)?
