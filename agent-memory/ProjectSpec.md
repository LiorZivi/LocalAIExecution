# Project Spec — LocalAIExecution

> **LocalAIExecution is a headless, scriptable, offline-capable platform for
> running AI generative models locally on your own GPU — built to be driven by
> automated, unattended pipelines, not a GUI.** It is the command-line /
> machine-readable counterpart to interactive tools like ComfyUI.

This is the **forward-looking purpose** of the project. For the original
implementation spec/plan see `plans\LocalAIExecution-spec.md` and
`plans\LocalAIExecution-plan.md` (historical). For *how* it's built, see
`human-docs\HighLevelArchitecture.md` (human reference).

---

## Why this exists

The goal is **AI generation you can call from a script, an agent, a CI job, or a
marketplace skill** — with no graphical UI, no running server, no cloud, and a
**stable, predictable contract** so automation can rely on it. You give it a
prompt and parameters; it returns a saved artifact and a machine-readable record,
every time, the same way.

GUI tools (ComfyUI) are excellent for *interactive* exploration and node-graph
tuning. They are awkward to drive unattended. LocalAIExecution fills the other
half: **the same local models, exposed as a clean automatable CLI.**

## Positioning

| | **LocalAIExecution** | **ComfyUI** | **Cloud APIs** |
|---|---|---|---|
| Interface | Headless CLI + `--json` | Interactive graph UI / server | HTTP API |
| Best for | Unattended/scripted/batch flows | Hands-on exploration & tuning | Zero local setup |
| Runs on | Your GPU, offline after first download | Your GPU | Someone else's servers |
| Cost / privacy | Free, fully local | Free, fully local | Paid, data leaves machine |
| Contract | Stable CLI + exit codes + provenance | Workflow JSON graphs | Vendor API |

They are **complementary**, not competing — both run the same local models on the
same GPU. Use ComfyUI to design a look; use LocalAIExecution to run it at scale,
unattended.

## What it does today

- **Text-to-image** via FLUX: `schnell` (fast, default) and `dev` (higher
  fidelity, gated). Implemented under `src\localai\capabilities\image\text_to_image\`.
- One-shot CLI (`localai generate "…"`) and an interactive REPL (load once, reuse).

## Where it's headed

More models and **modalities, each added as a self-contained adapter** with **no
changes to the core** (see `agent-memory\adding-a-capability.md`). The architecture
already groups capabilities by modality:

```
src\localai\capabilities\
├── image\   text_to_image (FLUX)        ← today
│            image_to_image, upscaling…   ← possible future image capabilities
├── video\   text_to_video (e.g. LTX-2)   ← a plausible next modality
└── audio\   …                            ← later
```

A new modality (e.g. **text-to-video**) is a new capability package + one import
line + (if it emits a new output type) one registered writer — exposing the
**same** scriptable `--json` contract as text-to-image. The hard part of a model
like LTX-2 is never the wiring; it's GPU memory management (quantization /
offload) and quality parity — not the platform seam.

## The automation contract (what makes it scriptable)

These are stable on purpose so a caller can depend on them:

1. **One-shot output** — a `generate`-style command prints the **saved absolute
   path** as its final stdout line.
2. **`--json` mode** — emits exactly **one JSON object** on stdout (capability,
   model, artifacts[], full provenance); all diagnostics go to stderr.
3. **Deterministic exit codes** — `0` ok · `1` unexpected · `2` invalid args ·
   `3` CUDA/torch wrong build · `4` GPU absent · `5` OOM · `6` gated/token ·
   `7` network/download · `8` unknown capability/model. (Full detail in
   `human-docs\skill-invocation.md`.)
4. **Predictable, collision-safe output paths** + a `.json` provenance sidecar
   beside every artifact.
5. **Reproducibility** — same prompt + seed + settings reproduces the result
   (within bf16 nondeterminism); the seed is always recorded.
6. **Offline after first download** — models cache locally (`HF_HUB_CACHE`); no
   cloud calls at generation time.
7. **GPU-first, no silent CPU fallback** — `localai doctor` gates the stack.

## Design principles

- **Modality-agnostic core, pluggable adapters.** The core (`src\localai\core`)
  carries no model/image/video specifics; capabilities plug in.
- **Stable surface over convenience.** The CLI/`--json`/exit-code contract does
  not churn, because automation breaks when contracts move.
- **Provenance everywhere.** Every artifact is self-describing and reproducible.
- **Honest about hardware.** Verify CUDA on the GPU; surface OOM/gated/network as
  actionable, coded errors — never a raw traceback.

## Non-goals

- No graphical or web UI; no dependency on a running server (e.g. ComfyUI).
- Not a ComfyUI replacement for interactive, node-graph experimentation.
- No cloud/paid inference.
- No new modality/capability built without an explicit ask (scope discipline).

## Example automated flows

- A content/marketing skill generates a batch of cover images overnight from a
  prompt list, reading each saved path from `--json`.
- An agent calls `localai generate "…" --json`, parses the artifact path, and
  attaches the image to a downstream step.
- A CI job regenerates reference images from fixed seeds and diffs them.
- (Future) a pipeline renders short videos from a script of prompts, unattended.

## Related docs

Agent-facing (read these):
- `agent-memory\adding-a-capability.md` — how to add a model/modality (no core edits).
- `agent-memory\STRUCTURE.md` — the repo's doc layout.

Human reference (in `human-docs\` — the agent normally skips these):
- `human-docs\HighLevelArchitecture.md` — core + adapters, and the end-to-end flow.
- `human-docs\FilesAndModelsStructure.md` — where everything lives (repo + model cache).
- `human-docs\skill-invocation.md` — the exact machine-readable contract + exit codes.
- `human-docs\validation.md` — measured end-to-end results against the success criteria.
- `AGENTS.md` — maintainer orientation.
