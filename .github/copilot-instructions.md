# Copilot instructions — LocalAIExecution

This repo is an **already-built**, generic **local-AI execution platform**: a
headless, scriptable, offline-capable platform for running AI generative models
locally on the GPU, **built to be driven by automated/unattended pipelines** (the
CLI / `--json` counterpart to GUI tools like ComfyUI). It is a reusable,
modality-agnostic core/runtime plus pluggable capability adapters.
These instructions are **project-wide**. Modality- or path-specific guidance
lives in scoped files under `.github\instructions\` (applied automatically by
path):

- `.github\instructions\docs.instructions.md` — writing docs under `docs\`.
- `.github\instructions\image-capability.instructions.md` — the image capability
  group under `src\localai\capabilities\image\`.

**Read these first** (they are the source of truth — keep them up to date):
- `docs\ProjectSpec.md` — the project's purpose + the stable automation contract.
- `AGENTS.md` — maintainer orientation.
- `docs\HighLevelArchitecture.md` — core + adapters and the end-to-end flow.
- `docs\FilesAndModelsStructure.md` — where everything lives (repo + model cache).
- `docs\adding-a-capability.md` — how to add a model/modality (no core edits).
- `docs\skill-invocation.md` — the machine-readable `--json` contract + exit codes.
- `docs\validation.md` — measured end-to-end results.
- `README.md` — user-facing setup/usage.

`plans\LocalAIExecution-spec.md` / `plans\LocalAIExecution-plan.md` are the
historical intent + implementation record.

## Project-wide constraints

- **Architecture:** a reusable core (`src\localai\core`) plus pluggable capability
  adapters (`src\localai\capabilities\*`). The core is modality-agnostic — it must
  contain **no** model-, image-, or capability-specific logic. Capabilities are
  grouped by modality: `src\localai\capabilities\image\` today, with future
  siblings like `video\` / `audio\`. Adding a capability = a new adapter module +
  one import line in the relevant modality manifest (and a new modality = one line
  in `src\localai\capabilities\__init__.py`) — **never edit the core** to add one.
- **Target GPU: RTX 5090 (Blackwell, sm_120).** PyTorch **must** be the CUDA 12.8
  (cu128) build (`https://download.pytorch.org/whl/cu128`); standard PyPI/cu121
  wheels don't support sm_120. Verify the GPU stack with `localai doctor` before
  any model work — **no CPU fallback**. (Installed/verified: torch 2.11.0+cu128.)
- **Self-contained** Python using Hugging Face `diffusers` — **not** ComfyUI, no
  running-server dependency.
- **CLI contract is stable and scriptable** (a future skill depends on it):
  one-shot commands print the saved absolute artifact path as the final stdout
  line; the global `--json` flag emits exactly one provenance object on stdout
  (diagnostics go to stderr). Exit codes are deterministic — see
  `docs\skill-invocation.md`.
- **Models download to the Hugging Face cache** (outside the repo; default
  `%USERPROFILE%\.cache\huggingface\hub`, relocatable via `HF_HUB_CACHE` /
  `HF_HOME`), not into the project. After the first
  download the tool runs offline.
- **Build/test:** set up with `scripts\bootstrap.ps1`; run the GPU-free unit suite
  with `.venv\Scripts\python.exe -m pytest`. **Never commit** `.venv\`, `outputs\`,
  the Hugging Face cache, or any token.
- **Scope discipline:** don't add new modalities or capabilities without an
  explicit ask.
