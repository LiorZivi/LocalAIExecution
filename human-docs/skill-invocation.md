# Skill-Invocation Contract

This document is the **stable, machine-readable contract** a future skill/agent
relies on to call LocalAIExecution unattended. It does **not** implement a skill
or any packaging — it only specifies the interface.

## Invocation

Call a capability command with the global `--json` flag:

```powershell
localai generate "<prompt>" --json [--model schnell|dev] [--steps N] [--preset NAME] [--seed N] [--output-dir DIR]
```

- `--json` suppresses all human chatter on **stdout** and emits exactly **one
  JSON object**. Diagnostics/progress go to **stderr**.
- The process exit code is the source of truth for success/failure (see below).

## stdout JSON schema

On success (exit 0), stdout is a single JSON object:

```json
{
  "capability": "text-to-image",
  "model": "schnell",
  "artifacts": [
    {
      "path": "C:\\absolute\\path\\to\\image.png",
      "type": "image",
      "metadata": {
        "capability_id": "text-to-image",
        "model_id": "schnell",
        "model_repo": "black-forest-labs/FLUX.1-schnell",
        "seed": 1234,
        "timestamp": "2026-06-28T00:00:00+00:00",
        "load_seconds": 12.3,
        "generate_seconds": 2.1,
        "device": "cuda",
        "dtype": "bfloat16",
        "offload": "none",
        "library_versions": { "torch": "...", "diffusers": "...", "transformers": "...", "localai": "..." },
        "params": {
          "prompt": "...",
          "negative_prompt": null,
          "steps": 4,
          "guidance": 0.0,
          "width": 1024,
          "height": 1024,
          "max_sequence_length": 256,
          "batch": 1
        }
      }
    }
  ]
}
```

Field notes:

- `artifacts` is an **array** — `--batch N` yields N entries.
- `path` is **absolute** and points at the saved file. A matching `.json`
  sidecar (same stem) is written next to it.
- `metadata` equals the sidecar's content (the full provenance record).
- A consumer should read **only stdout** for the result and parse it as one JSON
  object; treat stderr as logs.

## Exit codes

| Code | Meaning |
|------|---------|
| 0 | success |
| 1 | unexpected error |
| 2 | invalid arguments (bad size/steps/preset) |
| 3 | CUDA/torch unavailable or wrong build |
| 4 | no NVIDIA GPU detected |
| 5 | CUDA out of memory |
| 6 | gated model / token denied |
| 7 | network / download failure |
| 8 | unknown capability or model |

On any non-zero exit, the actionable message is printed to **stderr** (never a
raw traceback).

## Discovery

A consumer can enumerate capabilities and models machine-readably:

```powershell
localai capabilities --json
```

```json
{ "capabilities": [ { "id": "text-to-image", "display_name": "...",
  "models": [ { "id": "schnell", "repo": "...", "default": true, "gated": false,
               "default_steps": 4, "supports_guidance": false }, ... ] } ] }
```

And verify the GPU stack:

```powershell
localai doctor --json
```

## Output & offline assumptions

- Output paths are **predictable and non-colliding**:
  `<timestamp>_<capability>_<model>_<slug>_seed<seed>_<NNN>.png`. Pass
  `--output-dir` to control the location.
- After the **first** model download, generation works **offline** from the
  local Hugging Face cache (no cloud calls).
- Authentication: the ungated default may require a one-time HF login depending
  on the model host's current gating; gated models (e.g. `dev`) always require a
  token (exit 6 otherwise).

## Stability

The `--json` object shape, the exit-code table, and the absolute-path output
contract are the cross-capability surface and are intended to remain stable as
new capabilities are added.
