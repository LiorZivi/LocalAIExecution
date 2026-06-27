# LocalAIExecution — Agent Context

> **Read this file first, then read `LocalAIExecution-plan.md` and `LocalAIExecution-spec.md` (both in this directory) before doing anything.** This repo is being built from a reviewed implementation plan. Your job is to execute that plan.

This document preserves the full context from the planning session (held 2026-06-28) so a fresh Copilot CLI session opened in this directory starts fully informed.

## What this project is

A dedicated **local-AI-model execution platform** that runs AI models on the user's own GPU with **no cloud services**. It is structured as a reusable, modality-agnostic **core/runtime** plus pluggable **capability adapters**. The **first and only capability built now is text-to-image** (FLUX). Adding a future model later = a new adapter module + one registry entry, with **no changes to the core**.

This repo is standalone (it was planned inside the `ZiviDevelopmentMarketplace` plugin repo but lives on its own). It will later be published to GitHub and, later still, be invoked by a marketplace skill — so the CLI contract must stay **stable and scriptable**.

## Source of truth

- **`LocalAIExecution-spec.md`** — product-level intent, success criteria, non-goals, constraints.
- **`LocalAIExecution-plan.md`** — the phased implementation plan (**7 phases, 30 steps**). **Reviewed and PASSED at 9/10.** Execute it phase by phase; each phase ends in a working, testable state. Update the `[ ]` checkboxes to `[x]` as you complete steps.

## Confirmed decisions (do not re-litigate without asking the user)

- **Backend:** self-contained Python using Hugging Face `diffusers`, in its own virtual environment. **NOT ComfyUI** — no running-server dependency.
- **Default model:** `black-forest-labs/FLUX.1-schnell` — Apache-2.0 (ungated, no token), distilled for **~4 steps**, guidance 0, ignores negative prompts. This is the must-work path.
- **Optional higher-quality model:** `black-forest-labs/FLUX.1-dev` — **gated** (requires accepting the HF license + an HF token), benefits from **~20–50 steps**, uses real guidance (~3.5). Opt-in via flag.
  - Note: schnell is *distilled* for ≤4 steps — raising its step count does **not** improve quality. The "more steps = better" lever is FLUX.1-dev.
- **Interfaces (both required):**
  - **One-shot CLI:** `prompt → saved PNG → print the saved absolute path as the final stdout line`, plus a `--json` machine-readable mode (output path + provenance) for the future skill.
  - **Interactive REPL:** load the model **once** into VRAM, then generate per prompt with **no reload** (model load is ~10–40s; schnell generation is ~2s). Supports in-loop overrides and model switching with VRAM hygiene.
- **Architecture:** a `localai` package split into:
  - `localai.core` — GPU bootstrap + CUDA verification, capability/model **registry** (keyed by capability id + model id), layered **config** (global → per-capability → per-model precedence), generic **artifact + provenance** writer (PNG + sidecar JSON is the only concrete writer now), typed **errors** + deterministic exit codes, the top-level **CLI dispatcher** with the shared `--json` contract, and the resident **engine** (load once, reuse).
  - `localai.capabilities.text_to_image` — the first adapter, holding all FLUX-specific behavior.
- **Scope discipline:** text-to-image **only** is implemented now. Build the seams for other models, but do **NOT** implement img2img / inpainting / upscaling / video / audio / language / LoRA / GUI.

## Environment (probed 2026-06-28 on this machine)

- OS: **Windows**, PowerShell. Use Windows paths (backslashes).
- GPU: **NVIDIA RTX 5090, 32 GB, Blackwell (compute capability sm_120 / 12.0)**, driver 591.86 (observed value — yours may differ slightly; not a hard requirement).
- Python **3.12.10** on PATH. **`torch` is NOT installed.** No ComfyUI running (port 8188 closed).
- **CRITICAL — Blackwell / cu128:** Blackwell sm_120 is only supported by PyTorch wheels built for **CUDA 12.8 (cu128) or newer**. Install torch from `https://download.pytorch.org/whl/cu128` (fall back to the **cu128 nightly** index if the stable wheel lacks sm_120). Standard PyPI / cu121 wheels will fail or silently fall back to **CPU**. **Before any model work, verify** `torch.cuda.is_available()` is True, the device name contains "RTX 5090", and `torch.version.cuda` starts with `12.8`. The plan retires this risk first (Phase 2) behind a hard gate.

## Leanings on the spec's open questions

- **Default output directory:** repo-local `outputs\` (git-ignored), revisitable later.
- **Image size:** FLUX expects dimensions in **multiples of 16**; default 1024×1024, with square / widescreen / portrait presets.
- **Gated dev model:** wire schnell end-to-end first; add dev as a thin opt-in afterward.

## How to proceed in the next session

1. Read `LocalAIExecution-plan.md` and `LocalAIExecution-spec.md` fully.
2. Execute Phase 1 → 7 in order. **Do not skip the Phase 2 GPU/cu128 gate** — it is the make-or-break risk.
3. Keep the CLI contract stable: one-shot prints the absolute image path as the final stdout line; `--json` for machine consumers.
4. Update the plan's `[ ]` checkboxes to `[x]` as you complete each step.
