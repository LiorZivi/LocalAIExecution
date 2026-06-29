# End-to-End Validation

Records the spec's success criteria checked against the real environment
(Windows, NVIDIA RTX 5090, Blackwell sm_120, 32 GB; Python 3.12.10).

## Environment (verified)

- `localai doctor` →
  - GPU: NVIDIA GeForce RTX 5090 (driver 591.86, compute 12.0, 32607 MiB)
  - torch 2.11.0+cu128, CUDA 12.8, **CUDA available = True**
  - device arch `sm_120` is in torch's arch list (no CPU fallback)
  - on-device matmul smoke: **OK** on `cuda:0`
- Unit suite (`pytest`): **54 passed** (GPU-free).

## Success criteria

| # | Criterion | Status | Evidence |
|---|-----------|--------|----------|
| 1 | Cold-start one-shot produces a valid image and prints its saved path | **VERIFIED** | `localai generate "a serene mountain lake at dawn ..."` saved `outputs\20260628-002200_text-to-image_schnell_..._seed1234_000.png` (1024×1024 RGB, 1.69 MB) + sidecar; absolute path printed as the final stdout line |
| 2 | Interactive mode loads once; each prompt reuses without reload | **VERIFIED** | resident run: 1 load, 3 generations (`LOADS_IN_CACHE 1`); unit test `test_engine.py` also proves load-once/unload |
| 3 | Default fast model generates in ~seconds/image post-load | **VERIFIED** | schnell 1024×1024, 4 steps, `offload=model`: load ~5 s; first gen ~35 s (CUDA warmup); **subsequent resident gens ~10 s** |
| 4 | dev model + higher steps → visibly higher fidelity (slower) | DESIGN-VERIFIED | dev wired (guidance 3.5, ~28 steps, true-CFG negative prompt, max_seq 512); requires the user to accept the gated dev license to run live |
| 5 | Same prompt + seed + settings reproduces the same image | **VERIFIED** | seed 1234 produced byte-identical PNGs across two separate process runs (1,687,610 bytes each); resident run: seed 1 hash `7933ee2146a74943` reproduced exactly; distinct seeds gave distinct hashes |
| 6 | No cloud calls; works offline after caching | **VERIFIED** | only Hugging Face weight download on first run; the resident test and the second generation ran entirely from the local cache (~31 GB under `~/.cache/huggingface/hub`) |
| 7 | First-time user reaches an image via the README without GPU-lib troubleshooting | **VERIFIED** | `scripts/bootstrap.ps1` installs cu128 torch (torch 2.11.0+cu128), verifies CUDA on the RTX 5090, then warms schnell + runs a real generation smoke |

## Measured numbers (RTX 5090, 32 GB, schnell, 1024×1024, 4 steps)

| Mode | Result |
|------|--------|
| `offload=none` (full model > 32 GB → shared-memory spill) | **330 s** (unusable) |
| `offload=model` (default) — first generation | ~35 s (one-time CUDA kernel warmup) |
| `offload=model` (default) — warm resident generations | **~10–14 s** |
| Model load from local cache | ~5 s |
| Reproducibility (same seed) | byte-identical image |

## Error-path verification (no model download needed)

| Scenario | Expected | Status |
|----------|----------|--------|
| `localai doctor` on a healthy GPU | exit 0, RTX 5090 verified | VERIFIED |
| `localai capabilities` / `--json` | lists schnell (default) + dev | VERIFIED |
| `localai generate ... --steps 0` | exit 2 (invalid args), before any load | VERIFIED |
| `localai generate ... --width 500` (not multiple of 16) | exit 2 | VERIFIED |
| `localai generate ... --preset triangle` | exit 2 (unknown preset) | VERIFIED |
| `localai generate ... --model nope` | exit 8 (unknown model) | VERIFIED |
| schnell/dev without HF auth (current HF gating) | exit 6 with remediation | VERIFIED |

## Notes / deviations from the original plan

- **FLUX.1-schnell became login-gated on Hugging Face** after the plan was
  written. It remains Apache-2.0, but downloading now requires a one-time HF
  login (accept terms + read token). The tool surfaces missing auth as exit code
  6 with a clear remedy, and the token resolution path (`HF_TOKEN` /
  `hf auth login`) makes it work once authenticated.
- **`offload=none` does NOT fit a 32 GB card for FLUX** (the plan assumed it
  did). The full bf16 model is ~33 GB, so the text-to-image capability now
  **defaults to `offload=model`** (peak VRAM ~24 GB, ~10x faster than the
  shared-memory-spill path). `offload=none` remains available for cards with
  enough VRAM to hold the whole model.
