# LocalAIExecution — Maintainer Context

> Orientation for anyone (human or agent) working on this **already-built**
> repo. For user-facing setup/usage see `README.md`; for the machine contract
> see `docs/skill-invocation.md`; to add a model see `docs/adding-a-capability.md`.
> `plans/LocalAIExecution-spec.md` and `plans/LocalAIExecution-plan.md` remain as
> the historical intent + implementation record (all 7 phases complete).

## What this project is

A dedicated **local-AI-model execution platform** that runs AI models on the
user's own GPU with **no cloud services**. It is a reusable, modality-agnostic
**core/runtime** (`src\localai\core`) plus pluggable **capability adapters**
(`src\localai\capabilities\*`). The **first and only capability is text-to-image**
(FLUX). Adding a future model = a new adapter module + one import line in
`src\localai\capabilities\__init__.py`, with **no changes to the core**.

Standalone repo, published to GitHub; intended to later be invoked by a
marketplace skill — so the CLI contract (one-shot prints the saved absolute path
as the final stdout line; `--json` emits one provenance object) must stay
**stable and scriptable**.

## Architecture

- `src\localai\core` — GPU bootstrap + CUDA/sm_120 verification (`gpu.py`, `doctor`),
  capability/model **registry** keyed by `(capability_id, model_id)`, layered
  **config** (CLI > env `LOCALAI_*` > file[model>cap>global] > builtin), the
  resident **engine** (load once, reuse, `unload` for VRAM hygiene), generic
  **artifact + provenance** writer (collision-safe filenames + sidecar JSON),
  typed **errors** with deterministic exit codes, and the **CLI dispatcher** with
  the shared `--json` contract.
- `src\localai\capabilities\text_to_image` — the FLUX adapter (schnell/dev specs,
  size presets, PNG writer, `generate` one-shot, interactive REPL). The PNG +
  sidecar-JSON writer is the only concrete artifact writer.

**Scope discipline:** text-to-image **only**. The seams for other models exist,
but do **NOT** implement img2img / inpainting / upscaling / video / audio /
language / LoRA / GUI without an explicit ask.

## Models (current reality)

- Default `black-forest-labs/FLUX.1-schnell` — Apache-2.0, distilled for **~4
  steps**, guidance 0, ignores negative prompts. **Login-gated on Hugging Face:**
  a one-time HF token is needed to *download* the weights (`HF_TOKEN` or
  `hf auth login`); generation afterwards is fully local/offline. Missing auth
  surfaces as **exit code 6**.
- Optional `black-forest-labs/FLUX.1-dev` — **gated** (accept license + token),
  guidance ~3.5, ~20–50 steps (default 28), optional negative prompt via true
  CFG. Opt-in via `--model dev`.
- Note: schnell is *distilled* for ≤4 steps — more steps do **not** improve it.
  The "more steps = better" lever is dev.

## Environment & critical constraints

- OS: **Windows**, PowerShell. Use Windows paths (backslashes).
- GPU: **NVIDIA RTX 5090, 32 GB, Blackwell (sm_120 / cap 12.0)**.
- **Blackwell / cu128 (make-or-break):** sm_120 needs PyTorch built for **CUDA
  12.8+**. Installed via `scripts/bootstrap.ps1` from
  `https://download.pytorch.org/whl/cu128` (nightly cu128 as fallback). Standard
  PyPI / cu121 wheels fail or silently fall back to CPU. Verified working:
  **torch 2.11.0+cu128**, `cuda.is_available()` True, sm_120 in the arch list.
  Run `localai doctor` to re-verify the GPU stack.
- **VRAM reality:** FLUX's full bf16 footprint (~33 GB) **exceeds** the 32 GB
  card. With `offload=none` it spills to shared memory (~330 s/image). The
  text-to-image capability therefore **defaults to `offload=model`** (peak VRAM
  ~24 GB). Do **not** force `offload=none` on a 32 GB card.

## Measured performance (RTX 5090, schnell, 1024×1024, 4 steps, offload=model)

- Model load from local cache ~5 s; first generation ~35 s (one-time CUDA kernel
  warmup); **warm resident generations ~10 s**. Same seed → byte-identical image.

## Build / test / run

- Setup: `scripts/bootstrap.ps1` (creates `.venv`, installs cu128 torch + the
  package, runs `doctor` + a real generation smoke).
- Tests: `.venv\Scripts\python.exe -m pytest` — **63 GPU-free unit tests** (~4 s).
- CLI entry point: `localai` (`localai.core.cli:main`). Try `localai doctor`,
  `localai capabilities`, `localai generate "<prompt>"`, `localai interactive`.
- Outputs land in repo-local `outputs\` (git-ignored). Never commit
  `.venv/`, `outputs/`, the HF cache, or any token.

## Exit codes

0 ok · 1 unexpected · 2 invalid args · 3 CUDA/torch wrong build · 4 GPU absent ·
5 OOM · 6 gated/token · 7 network/download · 8 unknown capability/model.
