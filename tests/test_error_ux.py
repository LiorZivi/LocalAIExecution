"""Table-driven error-UX: each failure maps to its exit code + actionable text."""

from __future__ import annotations

import pytest

from localai.core import errors

# (error class, expected exit code, a remedy snippet expected in the message)
TABLE = [
    (errors.InvalidArgumentError, 2),
    (errors.CudaUnavailableError, 3),
    (errors.GpuNotDetectedError, 4),
    (errors.OutOfMemoryError, 5),
    (errors.GatedModelError, 6),
    (errors.NetworkError, 7),
    (errors.UnknownModelError, 8),
]


@pytest.mark.parametrize("cls, code", TABLE)
def test_error_renders_with_code_and_message(cls, code, capsys):
    err = cls("something went wrong", remedy="do the fix")
    assert err.exit_code == code

    rc = errors.handle_error(err)
    assert rc == code

    rendered = capsys.readouterr().err
    assert "something went wrong" in rendered
    assert "do the fix" in rendered
    assert "Traceback" not in rendered  # never a raw traceback
