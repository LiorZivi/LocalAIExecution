"""Typed errors with deterministic exit codes.

Every failure mode the platform can surface maps to a stable exit code so a
future skill can react programmatically. ``handle_error`` renders an actionable
message (never a raw traceback) and returns the code for ``core.cli.main``.
"""

from __future__ import annotations

import sys
from typing import Optional, TextIO

# Deterministic, documented exit codes.
EXIT_SUCCESS = 0
EXIT_UNEXPECTED = 1
EXIT_INVALID_ARGS = 2
EXIT_CUDA_UNAVAILABLE = 3
EXIT_GPU_NOT_DETECTED = 4
EXIT_OUT_OF_MEMORY = 5
EXIT_GATED_ACCESS = 6
EXIT_NETWORK = 7
EXIT_UNKNOWN_MODEL = 8


class LocalAIError(Exception):
    """Base class for all expected, user-facing failures.

    Carries a stable ``exit_code`` and an optional ``remedy`` describing the
    concrete next steps the user can take.
    """

    exit_code: int = EXIT_UNEXPECTED

    def __init__(self, message: str, *, remedy: Optional[str] = None) -> None:
        super().__init__(message)
        self.message = message
        self.remedy = remedy

    def formatted(self) -> str:
        """Human-readable rendering: an ``error:`` line plus optional remedy."""
        text = f"error: {self.message}"
        if self.remedy:
            text += f"\n  remedy: {self.remedy}"
        return text


class InvalidArgumentError(LocalAIError):
    """Bad CLI arguments / invalid values (size, steps, ...)."""

    exit_code = EXIT_INVALID_ARGS


class CudaUnavailableError(LocalAIError):
    """torch cannot use CUDA, or the installed build lacks this GPU's arch."""

    exit_code = EXIT_CUDA_UNAVAILABLE


class GpuNotDetectedError(LocalAIError):
    """No NVIDIA GPU could be detected on the machine."""

    exit_code = EXIT_GPU_NOT_DETECTED


class OutOfMemoryError(LocalAIError):
    """CUDA ran out of memory during load or generation."""

    exit_code = EXIT_OUT_OF_MEMORY


class GatedModelError(LocalAIError):
    """A gated model was requested without accepted license / valid token."""

    exit_code = EXIT_GATED_ACCESS


class NetworkError(LocalAIError):
    """A model download or network operation failed."""

    exit_code = EXIT_NETWORK


class UnknownModelError(LocalAIError):
    """Unknown / invalid capability or model id."""

    exit_code = EXIT_UNKNOWN_MODEL


def handle_error(exc: BaseException, *, stream: Optional[TextIO] = None) -> int:
    """Render *exc* to *stream* (stderr by default) and return its exit code.

    Known ``LocalAIError`` instances print their actionable message; anything
    else is an unexpected bug and maps to ``EXIT_UNEXPECTED``.
    """
    out = stream if stream is not None else sys.stderr
    if isinstance(exc, LocalAIError):
        print(exc.formatted(), file=out)
        return exc.exit_code
    print(f"error: unexpected failure: {exc}", file=out)
    return EXIT_UNEXPECTED
