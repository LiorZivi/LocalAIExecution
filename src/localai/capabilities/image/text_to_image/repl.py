"""Interactive REPL for the text-to-image capability (resident load-once).

Loads the configured model once via the core engine, then generates an image per
prompt with no reload. Supports per-prompt inline overrides, control commands
(``/set``, ``/show``, ``/model``, ``/help``), and model switching with VRAM
hygiene via the engine's ``unload``.
"""

from __future__ import annotations

import argparse
import shlex
import sys
from typing import Any, Dict, List, Optional, Tuple

from localai.core import registry
from localai.core.cli import emit_result
from localai.core.config import load_settings
from localai.core.engine import Engine
from localai.core.errors import EXIT_SUCCESS, InvalidArgumentError, LocalAIError
from localai.core.output import write_artifact

_SETTABLE = (
    "steps",
    "width",
    "height",
    "preset",
    "guidance",
    "negative_prompt",
    "seed",
    "batch",
    "output_dir",
)

_HELP = """\
commands:
  <prompt>                 generate an image with current settings
  <prompt> --steps N ...   generate with one-off inline overrides
  /set KEY VALUE           persist a setting (steps,width,height,preset,seed,
                           guidance,negative_prompt,batch,output_dir)
  /steps N | /seed N | /preset NAME | /size WxH   shortcuts for /set
  /model NAME              switch model (unloads the previous, frees VRAM)
  /show                    show current effective settings
  /help                    show this help
  /exit | /quit | Ctrl-D   leave (frees VRAM)
"""


class _RaisingArgumentParser(argparse.ArgumentParser):
    """argparse that raises instead of calling ``sys.exit`` on bad input."""

    def error(self, message: str) -> None:  # type: ignore[override]
        raise InvalidArgumentError(f"bad override: {message}")


def _inline_parser() -> _RaisingArgumentParser:
    p = _RaisingArgumentParser(add_help=False, prog="")
    p.add_argument("--steps", type=int)
    p.add_argument("--width", type=int)
    p.add_argument("--height", type=int)
    p.add_argument("--preset")
    p.add_argument("--guidance", type=float)
    p.add_argument("--negative-prompt", dest="negative_prompt")
    p.add_argument("--seed", type=int)
    p.add_argument("--batch", type=int)
    return p


def _parse_inline(line: str) -> Tuple[str, Dict[str, Any]]:
    """Split a prompt line into (prompt, one-off overrides)."""
    marker = line.find(" --")
    if marker == -1:
        return line.strip(), {}
    prompt = line[:marker].strip()
    flags = shlex.split(line[marker + 1 :])
    ns, _ = _inline_parser().parse_known_args(flags)
    overrides = {k: v for k, v in vars(ns).items() if v is not None}
    return prompt, overrides


def _print(json_mode: bool, message: str) -> None:
    if not json_mode:
        print(message, file=sys.stderr)


def _generate_once(
    adapter: Any,
    engine: Engine,
    capability_id: str,
    model_id: str,
    overrides: Dict[str, Any],
    config_path: Optional[str],
    json_mode: bool,
) -> None:
    spec = registry.get_model(capability_id, model_id)
    settings = load_settings(
        capability_id,
        model_id,
        cli_overrides=overrides,
        config_path=config_path,
        spec=spec,
        capability_defaults=getattr(adapter, "capability_defaults", None),
    )
    request = adapter.build_request(spec, settings)
    artifacts, record = engine.run(request)
    written = [
        write_artifact(a, record, settings, index=i) for i, a in enumerate(artifacts)
    ]
    _print(json_mode, f"done in {record.generate_seconds}s (seed {record.seed})")
    emit_result(json_mode, capability_id, model_id, written)


def run_repl(
    adapter: Any,
    engine: Engine,
    capability_id: str,
    model_id: str,
    initial_overrides: Dict[str, Any],
    json_mode: bool = False,
    config_path: Optional[str] = None,
) -> int:
    overrides: Dict[str, Any] = dict(initial_overrides)
    overrides.pop("model", None)  # model tracked separately

    spec = registry.get_model(capability_id, model_id)
    settings = load_settings(
        capability_id,
        model_id,
        cli_overrides=overrides,
        config_path=config_path,
        spec=spec,
        capability_defaults=getattr(adapter, "capability_defaults", None),
    )

    _print(json_mode, f"loading {capability_id}/{model_id} ({spec.repo}) ...")
    engine.load(capability_id, model_id, settings)
    _print(json_mode, "model loaded. type /help for commands, /exit to quit.")

    while True:
        try:
            line = input("localai> " if not json_mode else "")
        except EOFError:
            _print(json_mode, "")
            break
        except KeyboardInterrupt:
            _print(json_mode, "\n(interrupted — type /exit to quit)")
            continue

        line = line.strip()
        if not line:
            continue

        if line.startswith("/"):
            stop = _handle_command(
                line, adapter, engine, capability_id, model_id, overrides, json_mode
            )
            if isinstance(stop, str):  # model switch returns the new model id
                model_id = stop
                spec = registry.get_model(capability_id, model_id)
            elif stop is True:
                break
            continue

        prompt, inline = _parse_inline(line)
        if not prompt:
            _print(json_mode, "empty prompt; type /help")
            continue
        per_call = {**overrides, **inline, "prompt": prompt}
        try:
            _generate_once(
                adapter, engine, capability_id, model_id, per_call, config_path, json_mode
            )
        except LocalAIError as exc:
            _print(json_mode, exc.formatted())
        except Exception as exc:  # noqa: BLE001 - keep the REPL alive
            _print(json_mode, f"error: {exc}")

    engine.unload_all()
    return EXIT_SUCCESS


def _handle_command(
    line: str,
    adapter: Any,
    engine: Engine,
    capability_id: str,
    model_id: str,
    overrides: Dict[str, Any],
    json_mode: bool,
):
    parts = line[1:].split()
    cmd = parts[0].lower() if parts else ""
    rest = parts[1:]

    if cmd in ("exit", "quit"):
        return True
    if cmd == "help":
        _print(json_mode, _HELP)
        return None
    if cmd == "show":
        spec = registry.get_model(capability_id, model_id)
        eff = load_settings(
            capability_id,
            model_id,
            cli_overrides=overrides,
            spec=spec,
            capability_defaults=getattr(adapter, "capability_defaults", None),
        )
        from localai.capabilities.image.text_to_image.sizes import resolve_size

        try:
            width, height = resolve_size(
                eff.get_str("preset"), eff.get_int("width"), eff.get_int("height"), spec
            )
        except Exception:
            width = height = "?"
        _print(
            json_mode,
            f"model={model_id} steps={eff.get('steps')} "
            f"size={width}x{height} preset={eff.get('preset')} "
            f"seed={eff.seed} guidance={eff.get('guidance')} offload={eff.offload} "
            f"batch={eff.batch} output_dir={eff.output_dir}",
        )
        return None
    if cmd == "model":
        if not rest:
            _print(json_mode, "usage: /model NAME")
            return None
        new_model = rest[0]
        try:
            registry.get_model(capability_id, new_model)
        except LocalAIError as exc:
            _print(json_mode, exc.formatted())
            return None
        if new_model == model_id:
            _print(json_mode, f"already on model {new_model}")
            return None
        cap_defaults = getattr(adapter, "capability_defaults", None)
        _print(json_mode, f"switching {model_id} -> {new_model} (freeing VRAM) ...")
        engine.unload(capability_id, model_id)
        spec = registry.get_model(capability_id, new_model)
        eff = load_settings(
            capability_id, new_model, cli_overrides=overrides, spec=spec,
            capability_defaults=cap_defaults,
        )
        try:
            engine.load(capability_id, new_model, eff)
        except LocalAIError as exc:
            _print(json_mode, exc.formatted())
            # Recovery: the previous model was already unloaded; reload it so the
            # resident session keeps working instead of being left with nothing.
            _print(json_mode, f"restoring previous model {model_id} ...")
            try:
                prev_spec = registry.get_model(capability_id, model_id)
                prev_eff = load_settings(
                    capability_id, model_id, cli_overrides=overrides, spec=prev_spec,
                    capability_defaults=cap_defaults,
                )
                engine.load(capability_id, model_id, prev_eff)
                _print(json_mode, f"restored {model_id}.")
            except LocalAIError as exc2:
                _print(json_mode, f"could not restore {model_id}: {exc2.formatted()}")
            return None
        _print(json_mode, f"model {new_model} loaded.")
        return new_model

    # shortcut commands map to /set
    shortcuts = {"steps": "steps", "seed": "seed", "preset": "preset"}
    if cmd in shortcuts and rest:
        overrides[shortcuts[cmd]] = rest[0]
        _print(json_mode, f"set {shortcuts[cmd]}={rest[0]}")
        return None
    if cmd == "size" and rest and "x" in rest[0].lower():
        w, h = rest[0].lower().split("x", 1)
        overrides["width"], overrides["height"] = w, h
        overrides.pop("preset", None)
        _print(json_mode, f"set width={w} height={h}")
        return None
    if cmd == "set":
        if len(rest) < 2 or rest[0] not in _SETTABLE:
            _print(json_mode, f"usage: /set KEY VALUE  (keys: {', '.join(_SETTABLE)})")
            return None
        overrides[rest[0]] = " ".join(rest[1:])
        _print(json_mode, f"set {rest[0]}={overrides[rest[0]]}")
        return None

    _print(json_mode, f"unknown command '/{cmd}'; type /help")
    return None
