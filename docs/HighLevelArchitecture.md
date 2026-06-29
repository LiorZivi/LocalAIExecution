# High-Level Architecture

How LocalAIExecution is built, structured, and what happens end-to-end when you
run a command. For the on-disk layout (in-repo **and** the external model cache),
see [`FilesAndModelsStructure.md`](FilesAndModelsStructure.md).

---

## 1. The big idea: a core + pluggable adapters

The project is a **local-AI execution platform**, not just an image generator.
It separates *what is true for every model* (the **core/runtime**) from *what is
specific to one model family* (a **capability adapter**).

- **Core** (`localai.core`) — modality-agnostic. It knows nothing about FLUX,
  images, prompts, or steps. It knows about GPUs, a registry, layered config, a
  load-once engine, provenance, output files, errors, and the CLI.
- **Capability adapter** (`localai.capabilities.text_to_image`) — holds *all*
  FLUX-specific behavior behind one small interface.

Adding a future model (another image model, or audio/language later) is a **new
adapter module + one import line** — with **no changes to the core**. This is the
central design constraint.

```mermaid
flowchart TB
    user([User / future skill]) -->|localai ...| CLI

    subgraph core["localai.core  (reusable runtime — no model specifics)"]
        CLI["cli.py<br/>dispatcher + --json contract"]
        REG["registry.py<br/>(capability, model) lookup"]
        CFG["config.py<br/>layered settings precedence"]
        ENG["engine.py<br/>resident load-once engine"]
        GPU["gpu.py<br/>CUDA / sm_120 verification"]
        OUT["output.py + metadata.py<br/>filenames + provenance + writers"]
        ERR["errors.py<br/>typed errors + exit codes"]
        IFACE["interfaces.py<br/>CapabilityAdapter contract"]
    end

    subgraph cap["localai.capabilities.text_to_image  (the FLUX adapter)"]
        ADP["adapter.py<br/>load_pipeline + run"]
        MOD["models.py<br/>schnell / dev ModelSpecs"]
        SZ["sizes.py<br/>presets + validation"]
        WR["writer.py<br/>PNG + embedded params"]
        TCLI["cli.py / repl.py<br/>generate + interactive"]
    end

    CLI --> REG --> cap
    CLI --> CFG
    CLI --> ENG --> ADP
    ADP --> GPU
    ADP -->|FluxPipeline| HF["diffusers / transformers<br/>+ HF model cache"]
    ENG --> OUT
    cap -. implements .-> IFACE
    cap -. registers writer .-> OUT
    CLI --> ERR
```

---

## 2. Module responsibilities

| Module | Responsibility |
|--------|----------------|
| `core/cli.py` | Builds the top-level `localai` parser, registers core commands (`doctor`, `capabilities`), asks each adapter to contribute its subcommands, dispatches, and renders the shared `--json` result. Catches all errors → `handle_error`. |
| `core/interfaces.py` | The `CapabilityAdapter` protocol + `Artifact` / `InferenceRequest` base types. The only contract the core depends on. |
| `core/registry.py` | Holds capabilities and their `ModelSpec`s keyed by `(capability_id, model_id)`. `discover_capabilities()` imports the manifest so adapters self-register. |
| `core/config.py` | Resolves effective `Settings` with strict precedence (see §5). Type-coerces and validates values. |
| `core/engine.py` | The resident engine: selects device/dtype, **loads a pipeline once** and caches it by `(capability, model)`, routes `run`, and `unload`s to free VRAM. Wraps low-level failures (OOM) in typed errors and augments the provenance record with runtime fields. |
| `core/gpu.py` | `detect_nvidia_gpu()` (parses `nvidia-smi`), `verify_cuda()` (CUDA available + device + **sm_120 in torch's arch list**), a tiny on-device smoke, and the `doctor` report. The make-or-break Blackwell gate. |
| `core/metadata.py` | `ProvenanceRecord` — capability/model/repo, seed, timings, device/dtype/offload, library versions, and a capability-specific `params` block. Serializes to the sidecar JSON. |
| `core/output.py` | `register_writer(type, fn, ext)`, collision-safe `build_filename(...)`, and `write_artifact(...)` (selects the writer, writes the payload + `.json` sidecar). |
| `core/errors.py` | The exception hierarchy with stable exit codes + `handle_error()`. |
| `text_to_image/models.py` | The two `ModelSpec`s: `schnell` (default, ~4 steps, guidance 0) and `dev` (gated, ~28 steps, guidance ~3.5). |
| `text_to_image/adapter.py` | `load_pipeline` (build the FLUX pipeline on cuda/bf16 + offload) and `run` (seeded generator, schnell/dev-aware kwargs, timing, provenance). Maps HF/diffusers failures → typed errors. |
| `text_to_image/sizes.py` | Aspect presets + multiple-of-16 validation. |
| `text_to_image/writer.py` | The concrete `image` writer (PNG with provenance embedded as text chunks). |
| `text_to_image/cli.py` | The `generate` (one-shot) and `interactive` subcommands + arg→settings mapping. |
| `text_to_image/repl.py` | The resident REPL: load once, per-prompt overrides, `/set` `/model` `/show` commands, model switching with VRAM hygiene. |

---

## 3. The adapter contract

Every capability implements this small interface (`core/interfaces.py`). The
core is written against **only** this surface:

```python
class CapabilityAdapter(Protocol):
    capability_id: str
    display_name: str
    def list_models(self) -> list[ModelSpec]: ...
    def register_cli(self, subparsers, shared_parents) -> None: ...
    def build_request(self, model_spec, settings) -> InferenceRequest: ...
    def load_pipeline(self, model_spec, device, dtype, offload): ...
    def run(self, pipeline, request) -> tuple[list[Artifact], ProvenanceRecord]: ...
```

Self-registration happens on import: `text_to_image/adapter.py` ends with
`register_capability(TextToImageAdapter())`, and `capabilities/__init__.py`
imports the module. That single import line is the entire "plug-in" step.

---

## 4. End-to-end flow: `localai generate "a serene mountain lake at dawn"`

This is the full path from process start to the saved PNG + printed path.

```mermaid
sequenceDiagram
    autonumber
    actor U as User
    participant CLI as core/cli.py<br/>main()
    participant REG as core/registry.py
    participant H as text_to_image/cli.py<br/>_generate_handler
    participant CFG as core/config.py
    participant ENG as core/engine.py
    participant ADP as text_to_image/adapter.py
    participant GPU as core/gpu.py
    participant HF as diffusers + HF cache
    participant TORCH as FluxPipeline (GPU)
    participant OUT as core/output.py + writer.py
    participant FS as outputs\ on disk

    U->>CLI: localai generate "..."
    CLI->>CLI: build_parser()
    CLI->>REG: discover_capabilities()
    REG-->>CLI: text-to-image adapter self-registered
    CLI->>CLI: parse args → args.func = _generate_handler
    CLI->>H: args.func(args)

    H->>REG: get_model("text-to-image","schnell")
    REG-->>H: ModelSpec
    H->>CFG: load_settings(CLI > env > file > builtin)
    CFG-->>H: Settings (offload=model, steps=4, 1024x1024, ...)
    H->>ADP: build_request(spec, settings)
    Note over ADP: validates size (×16) and steps<br/>BEFORE any expensive load → exit 2 on bad args
    ADP-->>H: TextToImageRequest

    H->>ENG: load("text-to-image","schnell", settings)
    alt pipeline not cached
        ENG->>ADP: load_pipeline(spec, "cuda", bf16, "model")
        ADP->>GPU: verify_cuda()  (sm_120 gate, no CPU fallback)
        GPU-->>ADP: OK (RTX 5090, cu128)
        ADP->>ADP: _resolve_hf_token() (env or HF cache)
        ADP->>HF: FluxPipeline.from_pretrained(repo, bf16, token)
        Note over HF: first run downloads ~33 GB to the HF cache,<br/>afterwards loads from cache (offline)
        HF-->>ADP: pipeline (weights in CPU RAM)
        ADP->>ADP: enable_model_cpu_offload() + vae tiling
        ADP-->>ENG: pipeline
        ENG->>ENG: cache by (capability, model), record load time
    else already loaded (REPL / batch)
        ENG-->>H: reuse cached pipeline (no reload)
    end

    H->>ENG: run(request)
    ENG->>ADP: run(pipeline, request)
    ADP->>ADP: seed torch.Generator(cuda), assemble kwargs<br/>(schnell: guidance 0, max_seq 256)
    ADP->>TORCH: pipeline(**kwargs)
    loop denoising steps (≈4 for schnell)
        TORCH->>TORCH: transformer step on GPU<br/>(layers stream in via model offload)
    end
    TORCH-->>ADP: PIL image(s)
    ADP->>ADP: build ProvenanceRecord (seed, params, generate_seconds)
    ADP-->>ENG: [Artifact(image)], record
    ENG->>ENG: augment record (device, dtype, offload, lib versions)
    ENG-->>H: artifacts, record

    loop each artifact
        H->>OUT: write_artifact(artifact, record, settings)
        OUT->>OUT: build_filename() (timestamp_cap_model_slug_seed_NNN)
        OUT->>FS: write PNG (params embedded as text chunks)
        OUT->>FS: write .json sidecar (full provenance)
        OUT-->>H: {path, sidecar, type, metadata}
    end

    alt --json mode
        H->>U: one JSON object on stdout {capability, model, artifacts[]}
    else human mode
        H->>U: print each saved absolute path (final line = the image)
    end
    H-->>CLI: exit 0
    Note over CLI: any LocalAIError anywhere →<br/>handle_error() prints message to stderr +<br/>returns its exit code (no raw traceback)
```

### Interactive mode differs in one way
`localai interactive` runs the **load** step once, then loops the **run → write**
steps per prompt with the cached pipeline (no reload). `/model dev` calls
`engine.unload(...)` to free VRAM before loading the new pipeline, and restores
the previous model if the switch fails.

---

## 5. Configuration precedence

`load_settings(...)` merges layers; later layers win:

```
CLI args  >  env vars (LOCALAI_*)  >  config file
    ( [text-to-image.models.schnell] > [text-to-image] > [defaults] )
  >  built-in defaults ( model-spec > capability(offload=model) > core )
```

Example: `--steps 8` beats `LOCALAI_STEPS=6` beats a `localai.toml` value beats
the schnell default of `4`.

---

## 6. Errors & exit codes

Expected failures raise a typed `LocalAIError` subclass; `core/cli.main` catches
everything and renders an actionable message to **stderr** (never a raw
traceback), returning a deterministic code:

| 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| ok | unexpected | invalid args | CUDA/torch wrong build | GPU absent | OOM | gated/token | network/download | unknown capability/model |

These codes are part of the stable contract a skill relies on
(see [`skill-invocation.md`](skill-invocation.md)).

---

## 7. Why these choices

- **Core/adapter split** → new models don't touch tested core code.
- **Resident engine** → load the ~33 GB model once (~5 s from cache), then
  generate in ~10 s; the REPL and batch reuse one pipeline.
- **GPU gate first** → `verify_cuda()` refuses to silently run on CPU on the
  Blackwell sm_120 card; the cu128 wheel is mandatory.
- **Provenance everywhere** → every image is reproducible and self-describing
  (seed + settings in the PNG and the sidecar).
- **`--json` contract** → a future skill calls one command and reads one object.
