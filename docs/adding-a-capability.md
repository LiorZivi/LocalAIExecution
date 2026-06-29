# Adding a New Capability

The platform is split into a reusable **core** (`src\localai\core`) and pluggable
**capability adapters** (`src\localai\capabilities\*`). Adding a new model or even a
new modality is a **new adapter module + one manifest line** — with **no edits
to the core** and no changes to existing adapters.

This is the seam the text-to-image (FLUX) capability already uses; follow the
same shape.

## The contract: `CapabilityAdapter`

Implement the structural protocol in `src\localai\core\interfaces.py`:

```python
class CapabilityAdapter(Protocol):
    capability_id: str          # e.g. "text-to-speech"
    display_name: str

    def list_models(self) -> list[ModelSpec]: ...
    def register_cli(self, subparsers, shared_parents) -> None: ...
    def build_request(self, model_spec, settings) -> InferenceRequest: ...
    def load_pipeline(self, model_spec, device, dtype, offload): ...
    def run(self, pipeline, request) -> tuple[list[Artifact], ProvenanceRecord]: ...
```

You reuse the core for everything shared: the registry, layered config, the
resident engine (load-once + VRAM hygiene), provenance, collision-safe output,
typed errors/exit codes, and the `--json` contract.

## Steps

1. **Create the package** `src\localai\capabilities\<your_capability>\` with:
   - `models.py` — one `ModelSpec` per model (id, HF repo, pipeline class,
     defaults, gated flag, dtype, etc.).
   - `adapter.py` — your `CapabilityAdapter` implementation. At import time call
     `register_capability(YourAdapter())`.
   - `cli.py` — add your subcommand(s) in `register_cli`, mapping args to
     settings via `load_settings` (`src\localai\core\config.py`), driving `Engine`
     (`src\localai\core\engine.py`), and writing results with `write_artifact`
     (`src\localai\core\output.py`) + `emit_result` (`src\localai\core\cli.py`).
   - A writer (if you emit a new artifact type): implement a `write_fn` and call
     `register_writer("<type>", write_fn, "<ext>")` (`src\localai\core\output.py`).
     The core writes the `.json` sidecar for you.

2. **Register it (the one line)** — add an import to
   `src\localai\capabilities\__init__.py`:
   ```python
   from localai.capabilities import your_capability  # noqa: F401
   ```

That's it. The dispatcher auto-discovers your capability, lists it under
`localai capabilities`, and exposes your subcommands. **No core module changes.**

## Rules of thumb

- Keep heavy imports (torch, model libraries) **lazy** inside `load_pipeline` /
  `run` so registry/CLI discovery stays fast and GPU-free.
- Raise the **typed errors** from `core.errors` (or let the engine map CUDA OOM)
  so failures get the right exit code and an actionable message.
- Put model-specific knobs in your capability; keep the cross-capability surface
  (`--json`, output paths, exit codes) untouched.
- Return a `ProvenanceRecord` with your `params`; the engine fills in the shared
  fields (device, dtype, offload, load time, library versions).

## Reference

The text-to-image capability under
`src\localai\capabilities\text_to_image\` is a complete worked example:
`models.py`, `adapter.py`, `cli.py`, `repl.py`, `sizes.py`, `writer.py`, and the
single import line in `src\localai\capabilities\__init__.py`.
