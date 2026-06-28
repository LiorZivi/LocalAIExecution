"""GPU detection and CUDA verification.

The hardest platform risk is shipping a torch build that does not actually drive
the Blackwell (sm_120) RTX 5090. ``detect_nvidia_gpu`` reads ``nvidia-smi`` and
``verify_cuda`` confirms torch sees CUDA, the device, and that the GPU's arch is
in torch's compiled arch list — failing loudly (no silent CPU fallback).
"""

from __future__ import annotations

import shutil
import subprocess
from typing import Any, Dict, List, Optional

from localai.core.errors import CudaUnavailableError, GpuNotDetectedError


def detect_nvidia_gpu() -> Dict[str, Any]:
    """Probe the first NVIDIA GPU via ``nvidia-smi``.

    Returns ``{name, driver_version, compute_cap, memory_total_mib}``.
    Raises :class:`GpuNotDetectedError` if ``nvidia-smi`` is missing or fails.
    """
    exe = shutil.which("nvidia-smi")
    if not exe:
        raise GpuNotDetectedError(
            "nvidia-smi not found; no NVIDIA GPU/driver detected",
            remedy="install the NVIDIA driver and ensure nvidia-smi is on PATH",
        )
    try:
        out = subprocess.run(
            [
                exe,
                "--query-gpu=name,driver_version,compute_cap,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise GpuNotDetectedError(
            f"nvidia-smi failed: {exc}",
            remedy="verify the NVIDIA driver is installed and the GPU is visible",
        )
    line = out.stdout.strip().splitlines()[0]
    name, driver, cap, mem = [c.strip() for c in line.split(",")]
    return {
        "name": name,
        "driver_version": driver,
        "compute_cap": cap,
        "memory_total_mib": int(float(mem)) if mem else None,
    }


def _arch_token(capability: tuple[int, int]) -> str:
    return f"sm_{capability[0]}{capability[1]}"


def verify_cuda(expected_device_substr: Optional[str] = None) -> Dict[str, Any]:
    """Verify torch can drive the GPU; raise :class:`CudaUnavailableError` if not.

    Checks: torch importable, ``cuda.is_available()``, and the device's
    ``sm_XY`` arch is present in ``torch.cuda.get_arch_list()`` (so the build
    actually targets this GPU rather than falling back to CPU).
    """
    try:
        import torch
    except Exception as exc:  # noqa: BLE001 - report any import failure clearly
        raise CudaUnavailableError(
            f"failed to import torch: {exc}",
            remedy="run scripts\\bootstrap.ps1 to install the cu128 torch build",
        )

    if not torch.cuda.is_available():
        raise CudaUnavailableError(
            "torch reports CUDA is NOT available (likely a CPU-only build)",
            remedy=(
                "reinstall torch from the cu128 index: pip install torch "
                "--index-url https://download.pytorch.org/whl/cu128"
            ),
        )

    device_name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    arch_list: List[str] = list(torch.cuda.get_arch_list())
    cuda_version = torch.version.cuda
    arch_token = _arch_token(capability)
    arch_supported = arch_token in arch_list

    if not arch_supported:
        raise CudaUnavailableError(
            f"installed torch ({torch.__version__}, CUDA {cuda_version}) does not "
            f"support this GPU's arch {arch_token}; arch list = {arch_list}",
            remedy=(
                "install a Blackwell-capable build from the cu128 index "
                "(or the cu128 nightly index if the stable wheel lacks sm_120)"
            ),
        )

    if expected_device_substr and expected_device_substr.lower() not in device_name.lower():
        # Informational mismatch — not a hard failure (hardware may differ).
        pass

    return {
        "available": True,
        "device_name": device_name,
        "capability": list(capability),
        "arch_token": arch_token,
        "arch_supported": arch_supported,
        "arch_list": arch_list,
        "cuda_version": cuda_version,
        "torch_version": torch.__version__,
    }


def cuda_smoke() -> Dict[str, Any]:
    """Run a tiny on-device tensor op to prove the runtime stack executes.

    Returns ``{ok, device, value}``; raises :class:`CudaUnavailableError` if the
    op cannot run on CUDA.
    """
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        raise CudaUnavailableError(f"failed to import torch: {exc}")

    if not torch.cuda.is_available():
        raise CudaUnavailableError("CUDA not available for smoke test")
    try:
        a = torch.randn(256, 256, device="cuda")
        b = torch.randn(256, 256, device="cuda")
        c = (a @ b).sum()
        torch.cuda.synchronize()
    except Exception as exc:  # noqa: BLE001
        raise CudaUnavailableError(
            f"on-device tensor op failed: {exc}",
            remedy="the cu128/sm_120 runtime stack is not executing on the GPU",
        )
    return {"ok": True, "device": str(c.device), "value": float(c.detach().cpu())}


def run_doctor(json_mode: bool = False) -> Dict[str, Any]:
    """Run GPU detection + CUDA verification + smoke; return a report dict.

    Raises the relevant typed error on failure (mapped to exit codes by the CLI).
    """
    report: Dict[str, Any] = {}
    report["gpu"] = detect_nvidia_gpu()
    report["cuda"] = verify_cuda(expected_device_substr="RTX 5090")
    report["smoke"] = cuda_smoke()
    report["ok"] = True
    return report
