# Copilot instructions — LocalAIExecution

This repo is an **already-built** local-AI execution platform. For orientation
read `AGENTS.md` (maintainer context); for user-facing details see `README.md`,
`docs/skill-invocation.md`, and `docs/adding-a-capability.md`.
`plans/LocalAIExecution-spec.md` / `plans/LocalAIExecution-plan.md` are the historical
intent + implementation record.

Key constraints (full detail in `AGENTS.md`):

- A **local-AI execution platform** (reusable core + pluggable model adapters);
  **text-to-image (FLUX) is the first and only capability**. Self-contained
  Python `diffusers` — **not** ComfyUI. Adding a model = new adapter module + one
  import line; **no core changes**. Don't build other modalities without an ask.
- Default model `black-forest-labs/FLUX.1-schnell` (~4 steps, guidance 0);
  optional gated `black-forest-labs/FLUX.1-dev` (~20–50 steps, guidance ~3.5).
  Both are **login-gated on Hugging Face** — a one-time HF token is needed to
  *download* weights (`HF_TOKEN` / `hf auth login`); generation is then local.
- Target GPU **RTX 5090 (Blackwell, sm_120)**: PyTorch **must** be the CUDA 12.8
  (cu128) build (`https://download.pytorch.org/whl/cu128`); run `localai doctor`
  to verify CUDA-on-GPU (no CPU fallback). Installed/verified: torch 2.11.0+cu128.
- **VRAM:** FLUX bf16 (~33 GB) exceeds the 32 GB card — the capability defaults
  to `offload=model`; don't force `offload=none`.
- CLI contract is **stable/scriptable**: one-shot prints the saved absolute path
  as the final stdout line; `--json` emits one provenance object on stdout.
- Build/test: setup via `scripts/bootstrap.ps1`; run `.venv\Scripts\python.exe -m
  pytest` (63 GPU-free tests). Never commit `.venv/`, `outputs/`, the HF cache,
  or tokens.
