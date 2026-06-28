"""Generic artifact output: writer-selection seam + collision-safe filenames.

The core owns filename construction and the sidecar-JSON companion. Concrete
payload writers (e.g. the PNG image writer) register themselves by artifact
type, so the core stays modality-agnostic.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable, Dict, Tuple

from localai.core.errors import InvalidArgumentError
from localai.core.interfaces import Artifact
from localai.core.metadata import ProvenanceRecord

# artifact type -> (write_fn, file_extension)
WriterFn = Callable[[Artifact, Path, ProvenanceRecord], None]
_writers: Dict[str, Tuple[WriterFn, str]] = {}

_SLUG_RE = re.compile(r"[^a-zA-Z0-9]+")


def register_writer(artifact_type: str, write_fn: WriterFn, extension: str) -> None:
    """Register a payload writer for *artifact_type* producing *extension*."""
    _writers[artifact_type] = (write_fn, extension.lstrip("."))


def get_writer(artifact_type: str) -> Tuple[WriterFn, str]:
    try:
        return _writers[artifact_type]
    except KeyError:
        known = ", ".join(sorted(_writers)) or "(none)"
        raise InvalidArgumentError(
            f"no writer registered for artifact type '{artifact_type}'",
            remedy=f"registered types: {known}",
        )


def slugify(text: str, max_len: int = 40) -> str:
    """Filesystem-safe slug from arbitrary text (e.g. a prompt)."""
    slug = _SLUG_RE.sub("-", text.strip().lower()).strip("-")
    if len(slug) > max_len:
        slug = slug[:max_len].rstrip("-")
    return slug or "artifact"


def _compact_timestamp(iso_ts: str) -> str:
    # 2026-06-28T02:39:11+00:00 -> 20260628-023911
    digits = re.sub(r"[^0-9]", "", iso_ts.split("+")[0])
    return f"{digits[:8]}-{digits[8:14]}" if len(digits) >= 14 else digits


def build_filename(
    output_dir: Path,
    record: ProvenanceRecord,
    *,
    slug: str | None = None,
    index: int = 0,
    ext: str = "png",
) -> Path:
    """Return a guaranteed-free path under *output_dir*.

    Stem = ``<timestamp>_<capability>_<model>[_<slug>]_seed<seed>_<index>`` with
    a numeric suffix bump if the path already exists (same-second/batch safety).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = _compact_timestamp(record.timestamp)
    seed = record.seed if record.seed is not None else "rand"
    parts = [ts, record.capability_id, record.model_id]
    if slug:
        parts.append(slugify(slug))
    parts.append(f"seed{seed}")
    stem = "_".join(str(p) for p in parts)

    counter = index
    while True:
        candidate = output_dir / f"{stem}_{counter:03d}.{ext}"
        if not candidate.exists():
            return candidate
        counter += 1


def write_artifact(
    artifact: Artifact,
    record: ProvenanceRecord,
    settings: Any,
    *,
    index: int = 0,
) -> Dict[str, Any]:
    """Write one artifact + its sidecar JSON; return a result descriptor.

    Returns ``{"path", "sidecar", "type", "metadata"}`` where ``path`` is the
    absolute artifact path and ``metadata`` is the record as a dict.
    """
    write_fn, ext = get_writer(artifact.type)
    output_dir = Path(getattr(settings, "output_dir", "outputs"))
    path = build_filename(
        output_dir, record, slug=artifact.suggested_slug, index=index, ext=ext
    )
    write_fn(artifact, path, record)

    sidecar = path.with_suffix(".json")
    sidecar.write_text(record.to_json(), encoding="utf-8")

    return {
        "path": str(path.resolve()),
        "sidecar": str(sidecar.resolve()),
        "type": artifact.type,
        "metadata": record.to_dict(),
    }
