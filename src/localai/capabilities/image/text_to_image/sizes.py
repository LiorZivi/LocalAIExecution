"""Aspect-ratio presets and FLUX dimension validation (image-specific).

FLUX expects width/height that are multiples of 16. Presets map friendly names
to concrete dimensions; explicit ``--width/--height`` override the preset and
are validated to the model's required multiple and a sane range.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

from localai.core.errors import InvalidArgumentError
from localai.core.registry import ModelSpec

# Friendly preset -> (width, height). All multiples of 16.
PRESETS: Dict[str, Tuple[int, int]] = {
    "square": (1024, 1024),
    "portrait": (768, 1344),
    "landscape": (1344, 768),
    "widescreen": (1344, 768),
    "wide": (1344, 768),
    "tall": (768, 1344),
}

_MIN_DIM = 256
_MAX_DIM = 2048


def _validate_dim(value: int, multiple: int, axis: str) -> int:
    if value < _MIN_DIM or value > _MAX_DIM:
        raise InvalidArgumentError(
            f"{axis} {value} out of range [{_MIN_DIM}, {_MAX_DIM}]",
            remedy=f"choose a {axis} between {_MIN_DIM} and {_MAX_DIM}",
        )
    if value % multiple != 0:
        raise InvalidArgumentError(
            f"{axis} {value} must be a multiple of {multiple}",
            remedy=f"round {axis} to the nearest multiple of {multiple}",
        )
    return value


def resolve_size(
    preset: Optional[str],
    width: Optional[int],
    height: Optional[int],
    model_spec: ModelSpec,
) -> Tuple[int, int]:
    """Resolve final (width, height) from preset / explicit dims / model default.

    Precedence: explicit width+height > preset > model default. Explicit dims
    are validated to the model's required multiple; invalid sizes raise the
    typed argument error (exit code 2).
    """
    multiple = model_spec.size_multiple

    if width is not None or height is not None:
        if width is None or height is None:
            raise InvalidArgumentError(
                "both --width and --height must be given together",
                remedy="pass both dimensions, or use a --preset instead",
            )
        return (
            _validate_dim(int(width), multiple, "width"),
            _validate_dim(int(height), multiple, "height"),
        )

    if preset:
        key = preset.strip().lower()
        if key not in PRESETS:
            known = ", ".join(sorted(PRESETS))
            raise InvalidArgumentError(
                f"unknown size preset '{preset}'",
                remedy=f"available presets: {known}",
            )
        return PRESETS[key]

    return (model_spec.default_width, model_spec.default_height)
