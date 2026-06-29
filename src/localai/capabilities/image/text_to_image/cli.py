"""CLI wiring for the text-to-image capability.

Contributes the ``generate`` (one-shot) and ``interactive`` (resident REPL)
subcommands, maps their args to layered settings, drives the core engine, writes
artifacts, and prints results via the shared ``--json`` contract.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any, Dict, List, Optional

from localai.core import registry
from localai.core.cli import emit_result
from localai.core.config import load_settings
from localai.core.engine import Engine
from localai.core.errors import EXIT_SUCCESS, InvalidArgumentError
from localai.core.output import write_artifact

# argparse dests mapped into the settings layer (None values are ignored).
_OVERRIDE_KEYS = (
    "prompt",
    "steps",
    "width",
    "height",
    "preset",
    "guidance",
    "negative_prompt",
    "max_sequence_length",
    "model",
    "output_dir",
    "seed",
    "dtype",
    "offload",
    "batch",
)


def register(adapter: Any, subparsers: argparse._SubParsersAction, shared_parents: List[Any]) -> None:
    """Add the ``generate`` and ``interactive`` subcommands."""
    gen = subparsers.add_parser(
        "generate",
        parents=shared_parents,
        help="generate an image from a prompt (one-shot)",
        description="Generate an image from a text prompt and save it to disk.",
    )
    gen.add_argument("prompt", help="the text prompt (quote multi-word prompts)")
    _add_image_args(gen)
    gen.set_defaults(func=lambda args: _generate_handler(adapter, args))

    inter = subparsers.add_parser(
        "interactive",
        parents=shared_parents,
        help="load the model once and generate many prompts (REPL)",
        description="Resident mode: load the model once, then generate per prompt.",
    )
    _add_image_args(inter)
    inter.set_defaults(func=lambda args: _interactive_handler(adapter, args))


def _add_image_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--steps", type=int, default=None, help="number of inference steps")
    parser.add_argument("--width", type=int, default=None, help="image width (multiple of 16)")
    parser.add_argument("--height", type=int, default=None, help="image height (multiple of 16)")
    parser.add_argument(
        "--preset",
        default=None,
        help="size preset: square|portrait|landscape|widescreen",
    )
    parser.add_argument(
        "--guidance", type=float, default=None, help="guidance scale (dev only)"
    )
    parser.add_argument(
        "--negative-prompt",
        dest="negative_prompt",
        default=None,
        help="negative prompt (dev only; ignored by schnell)",
    )
    parser.add_argument(
        "--max-sequence-length",
        dest="max_sequence_length",
        type=int,
        default=None,
        help="max prompt token length",
    )


def _collect_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    for key in _OVERRIDE_KEYS:
        value = getattr(args, key, None)
        if value is not None:
            overrides[key] = value
    return overrides


def _resolve_model_id(capability_id: str, requested: Optional[str]) -> str:
    if requested:
        registry.get_model(capability_id, requested)  # validates / raises
        return requested
    return registry.default_model(capability_id).model_id


def _log(json_mode: bool, message: str) -> None:
    if not json_mode:
        print(message, file=sys.stderr)


def _generate_handler(adapter: Any, args: argparse.Namespace) -> int:
    json_mode = bool(getattr(args, "json_mode", False))
    capability_id = adapter.capability_id
    model_id = _resolve_model_id(capability_id, getattr(args, "model", None))
    spec = registry.get_model(capability_id, model_id)

    settings = load_settings(
        capability_id,
        model_id,
        cli_overrides=_collect_overrides(args),
        config_path=getattr(args, "config_path", None),
        spec=spec,
        capability_defaults=getattr(adapter, "capability_defaults", None),
    )

    engine = Engine()
    request = adapter.build_request(spec, settings)  # validate cheap args before load
    _log(json_mode, f"loading {capability_id}/{model_id} ({spec.repo}) ...")
    engine.load(capability_id, model_id, settings)
    _log(json_mode, f"generating ({request.steps} steps, {request.width}x{request.height}) ...")
    artifacts, record = engine.run(request)

    written = [
        write_artifact(artifact, record, settings, index=i)
        for i, artifact in enumerate(artifacts)
    ]
    _log(json_mode, f"done in {record.generate_seconds}s (seed {record.seed})")
    emit_result(json_mode, capability_id, model_id, written)
    return EXIT_SUCCESS


def _interactive_handler(adapter: Any, args: argparse.Namespace) -> int:
    from localai.capabilities.image.text_to_image.repl import run_repl

    json_mode = bool(getattr(args, "json_mode", False))
    capability_id = adapter.capability_id
    model_id = _resolve_model_id(capability_id, getattr(args, "model", None))

    engine = Engine()
    return run_repl(
        adapter=adapter,
        engine=engine,
        capability_id=capability_id,
        model_id=model_id,
        initial_overrides=_collect_overrides(args),
        json_mode=json_mode,
        config_path=getattr(args, "config_path", None),
    )
