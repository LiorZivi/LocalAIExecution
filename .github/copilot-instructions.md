# Copilot instructions — LocalAIExecution

This repository is implemented from a reviewed plan. **Before doing anything, read `AGENTS.md` in the repo root, then `LocalAIExecution-plan.md` and `LocalAIExecution-spec.md`.**

Key constraints (full detail in `AGENTS.md`):

- A **local-AI execution platform** (reusable core + pluggable model adapters); **text-to-image (FLUX) is the first and only capability built now**. Self-contained Python `diffusers` — **not** ComfyUI.
- Default model `black-forest-labs/FLUX.1-schnell` (ungated, ~4 steps); optional gated `black-forest-labs/FLUX.1-dev` (~20–50 steps).
- Target GPU is an **RTX 5090 (Blackwell, sm_120)**: PyTorch **must** come from the CUDA 12.8 (cu128) wheel index (`https://download.pytorch.org/whl/cu128`); **verify CUDA-on-GPU before any model work** (no CPU fallback).
- Two interfaces: one-shot CLI (prints saved absolute path; `--json` mode) and an interactive REPL (load model once, reuse).
- Execute the plan **phase by phase** (Phase 2 cu128 GPU gate first); keep the CLI contract stable; update plan checkboxes as you go.
