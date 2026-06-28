"""Top-level CLI dispatcher and the shared ``--json`` result contract.

Builds the ``localai`` parser, registers the core ``doctor`` and ``capabilities``
subcommands, then asks each registered capability adapter to contribute its own
subcommands — so a new capability adds commands without touching this module.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List, Optional

from localai import __version__
from localai.core import registry
from localai.core.errors import EXIT_SUCCESS, LocalAIError, handle_error


def build_global_parent() -> argparse.ArgumentParser:
    """Flags shared by every subcommand (the stable cross-capability surface)."""
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument(
        "--json",
        action="store_true",
        dest="json_mode",
        help="emit a single machine-readable JSON object on stdout (skill mode)",
    )
    parent.add_argument(
        "--config",
        dest="config_path",
        default=None,
        help="path to a config TOML (defaults to ./localai.toml or ./config.toml)",
    )
    parent.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="print extra diagnostics to stderr",
    )
    return parent


def build_common_gen_parent() -> argparse.ArgumentParser:
    """Common generation knobs offered to capability subcommands."""
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument("--model", default=None, help="model id (e.g. schnell, dev)")
    parent.add_argument(
        "--output-dir", dest="output_dir", default=None, help="directory for outputs"
    )
    parent.add_argument(
        "--seed", type=int, default=None, help="reproducibility seed (int)"
    )
    parent.add_argument(
        "--dtype", default=None, help="compute dtype (bfloat16|float16|float32)"
    )
    parent.add_argument(
        "--offload",
        default=None,
        choices=["none", "model", "sequential"],
        help="CPU offload strategy for tighter VRAM",
    )
    parent.add_argument(
        "--batch", type=int, default=None, help="number of images per prompt"
    )
    return parent


def emit_result(
    json_mode: bool,
    capability_id: str,
    model_id: str,
    written: List[Dict[str, Any]],
) -> None:
    """Render a generation result.

    JSON mode: one object ``{capability, model, artifacts:[{path,type,metadata}]}``
    on stdout, nothing else. Human mode: each saved absolute path on its own
    line (the final stdout line is a saved path — the one-shot contract).
    """
    if json_mode:
        payload = {
            "capability": capability_id,
            "model": model_id,
            "artifacts": [
                {"path": w["path"], "type": w["type"], "metadata": w["metadata"]}
                for w in written
            ],
        }
        sys.stdout.write(json.dumps(payload) + "\n")
        return
    for w in written:
        print(w["path"])


def _doctor_handler(args: argparse.Namespace) -> int:
    from localai.core import gpu

    report = gpu.run_doctor()
    if getattr(args, "json_mode", False):
        sys.stdout.write(json.dumps(report) + "\n")
        return EXIT_SUCCESS
    g = report["gpu"]
    c = report["cuda"]
    s = report["smoke"]
    print(f"GPU:        {g['name']} (driver {g['driver_version']}, "
          f"compute {g['compute_cap']}, {g['memory_total_mib']} MiB)")
    print(f"torch:      {c['torch_version']} (CUDA {c['cuda_version']})")
    print(f"CUDA:       available=True device='{c['device_name']}' "
          f"cap={tuple(c['capability'])} arch={c['arch_token']} (supported)")
    print(f"smoke:      on-device matmul ok on {s['device']}")
    print("doctor:     OK - GPU stack verified")
    return EXIT_SUCCESS


def _capabilities_handler(args: argparse.Namespace) -> int:
    caps = registry.list_capabilities()
    if getattr(args, "json_mode", False):
        payload = {
            "capabilities": [
                {
                    "id": cap.capability_id,
                    "display_name": getattr(cap, "display_name", cap.capability_id),
                    "models": [
                        {
                            "id": spec.model_id,
                            "repo": spec.repo,
                            "default": spec.is_default,
                            "gated": spec.gated,
                            "default_steps": spec.default_steps,
                            "supports_guidance": spec.supports_guidance,
                        }
                        for spec in cap.list_models()
                    ],
                }
                for cap in caps
            ]
        }
        sys.stdout.write(json.dumps(payload) + "\n")
        return EXIT_SUCCESS

    if not caps:
        print("(no capabilities registered)")
        return EXIT_SUCCESS
    for cap in caps:
        print(f"{cap.capability_id} - {getattr(cap, 'display_name', '')}")
        for spec in cap.list_models():
            tags = []
            if spec.is_default:
                tags.append("default")
            if spec.gated:
                tags.append("gated")
            suffix = f" [{', '.join(tags)}]" if tags else ""
            print(f"    {spec.model_id:10s} {spec.repo}{suffix}")
    return EXIT_SUCCESS


def build_parser() -> argparse.ArgumentParser:
    global_parent = build_global_parent()
    common_gen_parent = build_common_gen_parent()

    parser = argparse.ArgumentParser(
        prog="localai",
        description="Local AI execution platform — run AI models on your own GPU.",
    )
    parser.add_argument(
        "--version", action="version", version=f"localai {__version__}"
    )
    subparsers = parser.add_subparsers(dest="command", metavar="<command>")

    doctor = subparsers.add_parser(
        "doctor",
        parents=[global_parent],
        help="verify the GPU + CUDA (cu128/sm_120) stack",
    )
    doctor.set_defaults(func=_doctor_handler)

    caps = subparsers.add_parser(
        "capabilities",
        parents=[global_parent],
        help="list registered capabilities and their models",
    )
    caps.set_defaults(func=_capabilities_handler)

    # Let each capability contribute its own subcommands.
    registry.discover_capabilities()
    for adapter in registry.list_capabilities():
        adapter.register_cli(subparsers, [global_parent, common_gen_parent])

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not getattr(args, "command", None):
        parser.print_help()
        return EXIT_SUCCESS

    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print("aborted", file=sys.stderr)
        return 1
    except LocalAIError as exc:
        return handle_error(exc)
    except Exception as exc:  # noqa: BLE001 - last-resort safety net
        return handle_error(exc)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
