# LocalAIExecution

A dedicated **local-AI model execution platform** — run AI models on your own
GPU with **no cloud services, no paid APIs, and no running server**. It is a
reusable, modality-agnostic **core/runtime** plus pluggable **capability
adapters**. The first and only capability built today is **text-to-image**
(FLUX), and it is structured so additional local models plug in as a new adapter
module + one registration line — with no changes to the core.

- **Self-contained** Python using Hugging Face `diffusers` (not ComfyUI).
- **GPU-first**: installs the correct CUDA 12.8 (cu128) PyTorch for Blackwell
  (RTX 5090, sm_120) and *verifies CUDA on the GPU* before any model work — no
  silent CPU fallback.
- **Two interfaces**: a one-shot CLI (prints the saved image path; `--json` for
  scripts/skills) and an interactive REPL (load once, generate many).

---

## Requirements

- **Windows** with an **NVIDIA GPU** (built and verified on an RTX 5090,
  Blackwell / sm_120, 32 GB) and a recent driver.
- **Python 3.12** on `PATH`.
- Disk space for model weights (FLUX.1-schnell is ~33 GB on first download) and
  network access for that first download (offline afterwards).
- A **Hugging Face account + token**: the FLUX repos are login-gated on Hugging
  Face. Accept the model terms once and log in (see *Authentication* below).

> **Why cu128?** Blackwell (sm_120) is only supported by PyTorch wheels built for
> CUDA 12.8 or newer. Standard PyPI / cu121 wheels fail or silently fall back to
> CPU. The bootstrap installs from `https://download.pytorch.org/whl/cu128`.

---

## Setup

One command creates the venv, installs the cu128 PyTorch build + the package,
verifies CUDA on the GPU, and warms the default model with a real generation
smoke test:

```powershell
./scripts/bootstrap.ps1
```

Useful flags:

```powershell
./scripts/bootstrap.ps1 -SkipSmoke     # set up + verify, but don't download a model
./scripts/bootstrap.ps1 -Nightly       # use the cu128 *nightly* index from the start
```

### Manual setup (equivalent)

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\localai.exe doctor          # verify CUDA on the GPU
```

### Authentication (one-time)

The FLUX repositories are login-gated on Hugging Face. Once:

1. Accept the terms on the model page(s):
   - <https://huggingface.co/black-forest-labs/FLUX.1-schnell> (default)
   - <https://huggingface.co/black-forest-labs/FLUX.1-dev> (optional, gated)
2. Log in so the token is cached locally:
   ```powershell
   .\.venv\Scripts\huggingface-cli.exe login    # paste a read token
   # or set an environment variable:
   $env:HF_TOKEN = "hf_xxx"
   ```

Verify the GPU stack at any time:

```powershell
localai doctor          # human-readable
localai doctor --json   # machine-readable
```

---

## Usage

### One-shot generation

```powershell
# Prints the saved absolute image path as the final stdout line.
localai generate "a serene mountain lake at dawn"

# Knobs:
localai generate "a neon city street" --steps 4 --preset widescreen --seed 42
localai generate "a portrait of a fox" --width 768 --height 1344 --output-dir outputs
localai generate "four variations" --batch 4
```

Each run writes a **PNG** plus a **`.json` sidecar** with full provenance, into
the output directory (default `outputs/`).

### Machine-readable mode (`--json`)

```powershell
localai generate "a red bicycle" --json
```

Emits exactly one JSON object on stdout (diagnostics go to stderr):

```json
{
  "capability": "text-to-image",
  "model": "schnell",
  "artifacts": [
    { "path": "C:\\...\\outputs\\20260628-...png", "type": "image", "metadata": { } }
  ]
}
```

See [`docs/skill-invocation.md`](docs/skill-invocation.md) for the full contract.

### Interactive mode (load once, generate many)

```powershell
localai interactive
```

```
localai> a calm zen garden
localai> a calm zen garden --steps 4 --seed 7      # one-off inline overrides
localai> /set steps 4                              # persist a setting
localai> /size 1344x768                            # or /preset widescreen
localai> /seed 1234
localai> /model dev                                # switch model (frees VRAM, reloads)
localai> /show                                     # show current settings
localai> /help
localai> /exit
```

The model loads **once**; each prompt reuses it (schnell generates in a few
seconds post-load). Switching models unloads the previous pipeline and frees
VRAM first.

---

## Models

| Model     | Repo                              | Steps    | Guidance | Negative prompt | License gate | Speed / quality |
|-----------|-----------------------------------|----------|----------|-----------------|--------------|-----------------|
| `schnell` (default) | `black-forest-labs/FLUX.1-schnell` | ~4 (distilled) | 0 (off) | ignored | Apache-2.0 (login-gated on HF) | Fastest; a few seconds/image |
| `dev`     | `black-forest-labs/FLUX.1-dev`    | ~20–50 (default 28) | ~3.5 | optional (true CFG) | **Gated, non-commercial** | Higher fidelity; slower |

> `schnell` is **distilled for ≤4 steps** — raising its step count does **not**
> improve quality. The "more steps = better" lever is `dev`.

### Using the gated `dev` model

1. Accept the license on <https://huggingface.co/black-forest-labs/FLUX.1-dev>.
2. Provide a token (env var or `huggingface-cli login`):
   ```powershell
   $env:HF_TOKEN = "hf_xxx"
   localai generate "an ornate library, volumetric light" --model dev --steps 28 --guidance 3.5
   ```
Without access, `--model dev` exits with code **6** and remediation steps.

---

## Configuration

Settings resolve with this **precedence (highest first)**:

```
CLI args  >  env vars (LOCALAI_*)  >  config file
    (per-model table > per-capability table > [defaults])
  >  built-in defaults (model-spec > capability > core)
```

- **Config file**: copy `config.example.toml` to `localai.toml` (repo-local,
  auto-detected) or pass `--config PATH`.
- **Env vars**: any key as `LOCALAI_<KEY>` — e.g. `LOCALAI_STEPS`,
  `LOCALAI_OUTPUT_DIR`, `LOCALAI_OFFLOAD`, `LOCALAI_SEED`.

See `config.example.toml` for the full layered shape and every key.

### Output, naming & reproducibility

- Files are written to the output directory (default `outputs/`) as
  `<timestamp>_<capability>_<model>_<slug>_seed<seed>_<NNN>.png` with a matching
  `.json` sidecar; filenames are collision-safe across same-second/batch runs.
- The PNG embeds the provenance params as text chunks; the sidecar holds the
  full record (seed, timings, device/dtype, library versions, params).
- **Reproducibility**: the same prompt + seed + settings reproduces the same
  image, *assuming the same library/model versions, hardware, and dtype* (bf16
  has minor nondeterminism). If you don't pass `--seed`, a random seed is drawn
  and **recorded** so you can reproduce it later.

### Performance & VRAM tuning

> **Important (32 GB cards):** FLUX's full bf16 footprint is ~33 GB — larger than
> a 32 GB RTX 5090. With `offload=none` it oversubscribes VRAM and spills to
> shared system memory, making generation **~10x slower** (observed ~330 s for 4
> steps). The text-to-image capability therefore **defaults to `offload=model`**,
> which keeps peak VRAM ~24 GB and runs fast. Don't force `offload=none` unless
> your card has enough VRAM for the whole model at once.

- **offload**: `model` (default here) → `sequential` (lowest VRAM) → `none`
  (only if the full model fits in VRAM). Measured on an RTX 5090 (32 GB), schnell
  1024×1024, 4 steps, `offload=model`: model load ~5 s; first generation ~35 s
  (one-time CUDA kernel warmup); **subsequent resident generations ~10 s**.
- **dtype**: `bfloat16` (default) is fastest on Blackwell; `float16` also works;
  `float32` doubles VRAM.
- VAE tiling is enabled to guard decode-time memory at large sizes.
- **Attention**: built-in PyTorch SDPA (no xformers — avoided for sm_120
  compatibility).

---

## Troubleshooting (exit codes)

The CLI never prints a raw traceback for expected failures; it prints an
actionable message and returns a deterministic exit code:

| Code | Meaning                              | What to do |
|------|--------------------------------------|------------|
| 0    | success                              | — |
| 1    | unexpected error                     | file an issue with the message |
| 2    | invalid arguments (size/steps/...)   | fix the flag; sizes must be multiples of 16 |
| 3    | CUDA/torch unavailable or wrong build| reinstall torch from the cu128 index; run `localai doctor` |
| 4    | no NVIDIA GPU detected               | install the driver; ensure `nvidia-smi` is on `PATH` |
| 5    | CUDA out of memory                   | `--offload model`/`sequential`, reduce `--width/--height/--batch` |
| 6    | gated model / token denied           | accept the license + set `HF_TOKEN` / `huggingface-cli login` |
| 7    | network / download failure           | check connectivity; after one download it works offline |
| 8    | unknown capability or model          | run `localai capabilities` to see valid ids |

Run `localai doctor` first whenever generation won't start — it isolates GPU/CUDA
problems from model problems.

---

## Project layout

```
src/localai/
  core/                 reusable runtime (no model specifics)
    gpu.py              GPU detection + CUDA/sm_120 verification
    registry.py         capability/model registry
    config.py           layered settings precedence
    engine.py           resident load-once engine
    metadata.py         provenance record
    output.py           artifact writer-selection + collision-safe filenames
    errors.py           typed errors + exit codes
    cli.py              top-level dispatcher + --json contract
    interfaces.py       the CapabilityAdapter contract
  capabilities/
    image/                image modality group (future: video/, audio/)
      text_to_image/      the FLUX adapter (first image capability)
scripts/bootstrap.ps1   GPU-aware setup (cu128 PyTorch)
config.example.toml     example layered configuration
docs/                   architecture, file/model layout, contracts, validation
tests/                  fast, GPU-free unit tests
```

For a deeper understanding see:
- [`docs/HighLevelArchitecture.md`](docs/HighLevelArchitecture.md) — how the
  core + adapters fit together, plus a sequence diagram of the full
  invocation-to-output flow.
- [`docs/FilesAndModelsStructure.md`](docs/FilesAndModelsStructure.md) — where
  every file lives, in-repo and out (the `.venv` libraries and where the FLUX
  model weights are cached on disk).

## Running the tests

The unit suite is fast and **does not require a GPU**:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

## Extending: add a new capability

Adding a model/capability is an **adapter module + one manifest line** — no core
edits. See [`docs/adding-a-capability.md`](docs/adding-a-capability.md). The
machine-readable contract a skill relies on is in
[`docs/skill-invocation.md`](docs/skill-invocation.md).

## License

MIT (tool code). Models carry their own licenses — FLUX.1-schnell is Apache-2.0
(login-gated on HF); FLUX.1-dev is gated/non-commercial. See [`LICENSE`](LICENSE).
