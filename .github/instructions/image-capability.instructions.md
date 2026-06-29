---
applyTo: "src/localai/capabilities/image/**"
---

# Image capability group (src\localai\capabilities\image\)

This package holds **image-generating** capabilities. `text_to_image` (FLUX) is
the first; future image capabilities (e.g. `image_to_image`, upscaling) are
**sibling packages here** and register via
`src\localai\capabilities\image\__init__.py`. Non-image modalities (video, audio)
belong in sibling modality folders under `src\localai\capabilities\`, never here.

## FLUX models (src\localai\capabilities\image\text_to_image\models.py)

- Default `schnell` (`black-forest-labs/FLUX.1-schnell`): distilled for ~4 steps,
  guidance 0, **ignores negative prompts**. Raising its step count does NOT
  improve quality.
- Optional `dev` (`black-forest-labs/FLUX.1-dev`): ~20–50 steps (default ~28),
  real guidance ~3.5, optional negative prompt via true CFG.
- **Both are login-gated on Hugging Face:** a one-time token (`HF_TOKEN` or
  `hf auth login`) is needed to *download* weights; generation is then
  local/offline. Missing or denied access must map to **exit code 6** with an
  actionable remedy.

## Image rules

- Width/height must be **multiples of 16**; offer presets (square / portrait /
  landscape / widescreen). Invalid sizes raise the typed argument error (exit 2).
- **VRAM:** FLUX's full bf16 footprint (~33 GB) exceeds the 32 GB RTX 5090, so the
  capability **defaults to `offload=model`** (peak ~24 GB, ~10 s warm). Do NOT
  default to `offload=none` — it spills to shared memory and is ~10x slower.
- The only artifact type here is `image` → PNG + `.json` sidecar, with the
  provenance params embedded as PNG text chunks
  (`src\localai\capabilities\image\text_to_image\writer.py`).
- Keep `capability_id = "text-to-image"` and the `generate` / `interactive`
  command names **stable** — they are part of the scriptable contract.
- Rely on PyTorch SDPA attention; **do not add xformers** (sm_120 incompatible).

## Don't

- Don't put image-specific logic in the core (`src\localai\core`) — it must stay
  modality-agnostic.
- Don't implement new image capabilities (image_to_image, inpainting, upscaling,
  ...) without an explicit ask.
