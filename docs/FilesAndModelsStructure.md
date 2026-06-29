# Files & Models Structure

Where everything lives — **inside** the repository and **outside** it (the Python
runtime libraries and the multi-gigabyte model weights). Paths shown are from the
machine this was built on; substitute your own user folder where noted.
`%USERPROFILE%` is `C:\Users\<you>` (here: `C:\Users\USER`).

For how the pieces interact at runtime, see
[`HighLevelArchitecture.md`](HighLevelArchitecture.md).

---

## 1. Inside the repository (`C:\Dev\MyRepos\LocalAIExecution`)

```
LocalAIExecution/
├── AGENTS.md                     # maintainer orientation (read first)
├── README.md                     # user-facing setup + usage
├── LICENSE                       # MIT (tool code only)
├── pyproject.toml                # packaging, deps, `localai` entry point
├── config.example.toml           # example layered config (copy → localai.toml)
├── .gitignore                    # ignores .venv/, outputs/, caches, tokens
│
├── plans/                        # historical planning record
│   ├── LocalAIExecution-spec.md  #   product intent
│   └── LocalAIExecution-plan.md  #   phased implementation plan
│
├── scripts/
│   └── bootstrap.ps1             # GPU-aware setup: venv + cu128 torch + verify
│
├── src/localai/                  # the installable package (src-layout)
│   ├── __init__.py               # __version__
│   ├── core/                     # reusable runtime (no model specifics)
│   │   ├── cli.py                # top-level dispatcher + --json contract
│   │   ├── interfaces.py         # CapabilityAdapter protocol + Artifact
│   │   ├── registry.py           # (capability, model) registry
│   │   ├── config.py             # layered settings precedence
│   │   ├── engine.py             # resident load-once engine
│   │   ├── gpu.py                # CUDA / sm_120 verification + doctor
│   │   ├── metadata.py           # ProvenanceRecord
│   │   ├── output.py             # writers + collision-safe filenames
│   │   └── errors.py             # typed errors + exit codes
│   └── capabilities/
│       ├── __init__.py           # manifest: imports each modality group
│       └── image/                # image modality group (future siblings: video/, audio/)
│           ├── __init__.py       # imports each image capability
│           └── text_to_image/    # the FLUX adapter (first image capability)
│               ├── __init__.py   # imports adapter (self-registers)
│               ├── adapter.py    # load_pipeline + run + error mapping
│               ├── models.py     # schnell / dev ModelSpecs
│               ├── sizes.py      # presets + dimension validation
│               ├── writer.py     # PNG image writer (registers type "image")
│               ├── cli.py        # generate + interactive subcommands
│               └── repl.py       # resident interactive loop
│
├── tests/                        # 63 GPU-free unit tests (pytest)
│   ├── conftest.py               # GPU-free dummy adapter fixture
│   └── test_*.py                 # config, registry, engine, cli, sizes, ...
│
└── docs/
    ├── HighLevelArchitecture.md  # this file's sibling
    ├── FilesAndModelsStructure.md# (this file)
    ├── skill-invocation.md       # the machine --json contract
    ├── adding-a-capability.md    # how to plug in a new model
    └── validation.md             # measured end-to-end results
```

### Generated at runtime, but **git-ignored** (never committed)

| Path | What it is |
|------|-----------|
| `.venv/` | The Python virtual environment (~4.5 GB; see §2). |
| `outputs/` | Generated `*.png` images + `*.json` provenance sidecars. |
| `localai.toml` | Your local config (if you create one from the example). |
| `src/localai.egg-info/`, `**/__pycache__/`, `.pytest_cache/` | Build/test artifacts. |

The `.gitignore` also blocks `*.safetensors`, `*.ckpt`, `.env*`, and `*.token`
so weights and secrets can never be committed by accident.

---

## 2. The virtual environment — `…\LocalAIExecution\.venv\` (~4.5 GB)

Lives inside the repo folder but is **git-ignored**. It holds the Python
interpreter copy and every installed library. The heavyweight is PyTorch
(the cu128 CUDA runtime ships inside the wheel).

Each library installs as a folder under `.venv\Lib\site-packages\` — so `torch`
is at `.venv\Lib\site-packages\torch\`, `diffusers` at
`.venv\Lib\site-packages\diffusers\`, and so on for every row below.

| Library | Version (verified) | Role |
|---------|--------------------|------|
| `torch` | **2.11.0+cu128** | GPU tensor runtime; the CUDA 12.8 build that drives Blackwell sm_120. |
| `diffusers` | 0.38.0 | The `FluxPipeline` and the diffusion scheduler/loop. |
| `transformers` | 5.12.1 | The text encoders (CLIP + T5) used by FLUX. |
| `accelerate` | 1.14.0 | Device placement + the CPU-offload hooks. |
| `safetensors` | 0.8.0 | Fast, safe weight file format loader. |
| `huggingface_hub` | 1.21.0 | Downloads weights; resolves the auth token. |
| `pillow` (PIL) | 12.x | Builds and saves the PNG image. |
| `sentencepiece`, `protobuf` | — | T5 tokenizer dependencies. |
| `numpy` | 2.5.x | Array glue. |
| `pytest` | 9.x | The test runner (dev only). |

Key executables created in `.venv\Scripts\`: `python.exe`, **`localai.exe`** (the
CLI entry point), `hf.exe` (HuggingFace CLI), `pytest.exe`.

> The cu128 PyTorch wheel is **not** on PyPI and is **not** in `pyproject.toml`'s
> dependency list — `scripts/bootstrap.ps1` installs it explicitly from
> `https://download.pytorch.org/whl/cu128`. Installing the package without the
> bootstrap would leave you with no (or a wrong) torch.

---

## 3. The model weights — the Hugging Face cache (OUTSIDE the repo)

This is the answer to "where is the FLUX model downloaded?". Weights do **not**
live in the project — they go to the **Hugging Face model cache**, the directory
that directly holds the `models--…` folders, set by the `HF_HUB_CACHE`
environment variable. **This machine points it at**
`C:\AI\LocalModels\huggingface\`, so the model sits **directly** there with no
`hub\` wrapper. (If unset, the default is `%USERPROFILE%\.cache\huggingface\hub\`.
See §6 for the `HF_HUB_CACHE` vs `HF_HOME` choice.)

```
%HF_HUB_CACHE%\   ← C:\AI\LocalModels\huggingface\   (the model cache dir; no hub\ wrapper)
└── models--black-forest-labs--FLUX.1-schnell\   ← 31.4 GB (the model)
    ├── blobs\          # the actual file contents, named by hash
    ├── refs\           # branch (main) → commit-hash pointer
    └── snapshots\
        └── 741f7c3ce8b383c54771c7003378a50191e9efe9\   # the commit pulled
            ├── model_index.json     # which components make up the pipeline
            ├── scheduler\           # denoising scheduler config
            ├── transformer\         # the DiT — ~22.7 GB (3 shards)
            ├── text_encoder\        # CLIP text encoder
            ├── text_encoder_2\      # T5-XXL text encoder — ~8.9 GB (2 shards)
            ├── tokenizer\           # CLIP tokenizer
            ├── tokenizer_2\         # T5 tokenizer
            └── vae\                 # decodes latents → pixels
```

> On the next download HF recreates a small `.locks\` (empty) and a tiny
> `CACHEDIR.TAG` marker beside `models--…` — both harmless. Your **auth** (the
> `token` file) and the `xet\` download-accelerator cache live separately at the
> default HF home `%USERPROFILE%\.cache\huggingface\` (a few KB), keeping this
> model folder clean.

> **How the cache layout works:** files are stored once in `blobs\` (content-
> addressed by hash). A `snapshots\<commit>\` folder presents them in the normal
> component layout via symlinks (or copies on Windows without Developer Mode —
> hence the harmless "symlinks not supported" warning, which just uses more
> disk). `refs\main` records which commit the snapshot corresponds to.

### FLUX.1-schnell component sizes (≈ 31.4 GB total)

| Component | Size | What it does |
|-----------|------|--------------|
| `transformer\` (DiT, 3 safetensors shards) | **~22.7 GB** | The diffusion backbone that denoises the image latent. |
| `text_encoder_2\` (T5-XXL, 2 shards) | **~8.9 GB** | Turns the prompt into rich text embeddings. |
| `text_encoder\` (CLIP) | ~0.2 GB | A second, smaller prompt encoder. |
| `vae\` | ~0.1 GB | Decodes the final latent into RGB pixels. |
| schedulers / tokenizers / configs | tiny | Glue + JSON config. |

This ~33 GB bf16 footprint is **why the tool defaults to `offload=model`** —
it does not all fit in 32 GB of VRAM at once, so `accelerate` streams components
between system RAM and the GPU during generation.

### The optional `dev` model
`black-forest-labs/FLUX.1-dev` is **not** downloaded until you explicitly run
`--model dev` (and have accepted its gated license + provided a token). When you
do, it lands beside schnell as
`models--black-forest-labs--FLUX.1-dev\` in the same `HF_HUB_CACHE` dir (similar size).

---

## 4. Authentication files (OUTSIDE the repo)

| Path | What it is |
|------|-----------|
| `%USERPROFILE%\.cache\huggingface\token` | The active HF token (plain text), written by `hf auth login`. Kept at the default HF home (separate from the model cache) so the model folder stays clean. |
| `%USERPROFILE%\.cache\huggingface\stored_tokens` | Named tokens you've saved. |
| env `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` | Alternative: supply the token per-shell instead of logging in. |

The adapter's `_resolve_hf_token()` checks the env vars first, then this cache.
The token is needed **only to download** gated weights; once cached, generation
runs fully offline. **Never commit these files** (the repo's `.gitignore` guards
against it, but they live outside the repo anyway).

---

## 5. Quick reference — "where does X live?"

| Thing | Location | In repo? | In git? |
|-------|----------|----------|---------|
| Source code | `src\localai\` | ✅ | ✅ |
| Tests | `tests\` | ✅ | ✅ |
| Docs | `docs\` | ✅ | ✅ |
| CLI executable | `.venv\Scripts\localai.exe` | ✅ (in `.venv`) | ❌ |
| Python libraries (torch, diffusers…) | `.venv\Lib\site-packages\` | ✅ (in `.venv`) | ❌ |
| **FLUX model weights** | `C:\AI\LocalModels\huggingface\models--…\` (via `HF_HUB_CACHE`; default `%USERPROFILE%\.cache\huggingface\hub\`) | ❌ | ❌ |
| HF auth token | `%USERPROFILE%\.cache\huggingface\token` (default HF home) | ❌ | ❌ |
| Generated images + sidecars | `outputs\` | ✅ | ❌ |
| Your config | `localai.toml` (repo root) | ✅ | ❌ |

---

## 6. Relocating the model cache (and sharing a root with ComfyUI)

The Hugging Face model cache can be large. Two env vars control where it goes:

- **`HF_HUB_CACHE`** — the directory that *directly* holds the `models--…`
  folders. **This machine sets it to `C:\AI\LocalModels\huggingface`**, so the
  model sits at `C:\AI\LocalModels\huggingface\models--…` with **no `hub\`
  wrapper** — the cleanest layout.
- **`HF_HOME`** — relocates the *whole* HF home: its `hub\` model cache **plus**
  `token`, `stored_tokens`, and the `xet\` accelerator. Use this if you'd rather
  keep everything together; the model then lives under `…\hub\models--…` and the
  auth files sit alongside it.

With neither set, the default is `%USERPROFILE%\.cache\huggingface\` (models under
`…\hub\`). To set up the clean layout and move an existing download:

```powershell
# point the model cache at a clean folder (no hub\ wrapper), persistently
[Environment]::SetEnvironmentVariable('HF_HUB_CACHE','C:\AI\LocalModels\huggingface','User')
# move an existing download up out of its hub\ wrapper into that folder:
Move-Item "$env:USERPROFILE\.cache\huggingface\hub\models--*" "C:\AI\LocalModels\huggingface\"
```

`scripts\bootstrap.ps1 -ModelsDir C:\AI\LocalModels\huggingface` sets
`HF_HUB_CACHE` for you before the first download. Your tiny `token` and `xet\`
accelerator stay at the default HF home (a few KB), keeping the model folder clean.

### Sharing the root with ComfyUI

Goal: one root, `C:\AI\LocalModels\`, holding every local model — both this
project's and ComfyUI's:

```
C:\AI\LocalModels\
├── huggingface\   ← HF_HUB_CACHE (this project + any diffusers / transformers tool)
│   └── models--...\            (diffusers FLUX — sharded pipeline layout)
└── comfyui\       ← ComfyUI model categories (single-file layout)
    ├── checkpoints\  unet\  diffusion_models\  vae\  clip\  text_encoders\  loras\ ...
```

> **The two tools use different on-disk formats and cannot share the *same*
> files for a model like FLUX.** diffusers needs the HF *cache* layout
> (`models--org--name\snapshots\<hash>\{transformer,vae,text_encoder,...}\`,
> sharded); ComfyUI needs *single-file* `.safetensors` in category folders
> (`unet\flux1-schnell.safetensors`, `vae\ae.safetensors`,
> `clip\clip_l.safetensors`, `text_encoders\t5xxl_*.safetensors`). A model used by
> both is therefore stored once per format — but every model still lives under the
> one `C:\AI\LocalModels` root.

Point ComfyUI at the shared root with an `extra_model_paths.yaml` in your ComfyUI
folder (`C:\AI\ComfyUI\resources\ComfyUI\extra_model_paths.yaml`):

```yaml
local_models:
  base_path: C:/AI/LocalModels/comfyui/
  checkpoints: checkpoints/
  unet: unet/
  diffusion_models: diffusion_models/
  vae: vae/
  clip: clip/
  text_encoders: text_encoders/
  clip_vision: clip_vision/
  loras: loras/
  controlnet: controlnet/
  upscale_models: upscale_models/
  embeddings: embeddings/
```

(Alternatively, *move* ComfyUI's existing `…\ComfyUI\models` folder to
`C:\AI\LocalModels\comfyui` and set `base_path` to it.) ComfyUI keeps reading its
native single-file models from there; this project keeps reading the HF cache from
`huggingface\` — both under the one root.

**Advanced (optional, not recommended as default):** ComfyUI can load a
diffusers-format model via its `DiffusersLoader` node (`models\diffusers\`), so in
principle it could reuse this project's HF copy. For FLUX the native single-file
loaders are the common, better-supported path, so true one-copy sharing is usually
more trouble than a second download.

---

## 7. Total disk footprint

| Item | Approx size |
|------|-------------|
| The repo source (code + docs + tests) | < 1 MB |
| `.venv\` (Python + torch/cu128 + libs) | ~4.5 GB |
| FLUX.1-schnell weights (HF cache) | ~31.4 GB |
| FLUX.1-dev weights (only if used) | ~32 GB (optional) |
| Each generated 1024×1024 PNG | ~1.6 MB + a ~1 KB sidecar |

---

## 8. Clean up / uninstall

- Remove generated images: delete the repo-local `outputs\` folder.
- Remove the environment: delete `.venv\`.
- **Reclaim the most space** — delete the downloaded model:
  `C:\AI\LocalModels\huggingface\models--black-forest-labs--FLUX.1-schnell\`
  (the model cache dir is `%HF_HUB_CACHE%`; default `%USERPROFILE%\.cache\huggingface\hub\`).
  Delete the `dev` folder too if present. They re-download on next use.
- Forget the token: `hf auth logout` (or delete the `token` file).
