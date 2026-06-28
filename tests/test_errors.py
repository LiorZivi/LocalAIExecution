"""Typed errors carry their documented exit code and an actionable message."""

from __future__ import annotations

import pytest

from localai.core import errors

EXIT_CASES = [
    (errors.InvalidArgumentError, 2),
    (errors.CudaUnavailableError, 3),
    (errors.GpuNotDetectedError, 4),
    (errors.OutOfMemoryError, 5),
    (errors.GatedModelError, 6),
    (errors.NetworkError, 7),
    (errors.UnknownModelError, 8),
]


@pytest.mark.parametrize("cls, code", EXIT_CASES)
def test_exit_codes(cls, code):
    assert cls("msg").exit_code == code


def test_handle_known_error_prints_message_and_remedy(capsys):
    code = errors.handle_error(errors.GatedModelError("denied", remedy="set HF_TOKEN"))
    assert code == 6
    err = capsys.readouterr().err
    assert "error: denied" in err
    assert "remedy: set HF_TOKEN" in err


def test_handle_unexpected_error(capsys):
    code = errors.handle_error(ValueError("boom"))
    assert code == errors.EXIT_UNEXPECTED == 1
    assert "unexpected" in capsys.readouterr().err


def test_all_exit_codes_distinct():
    codes = {code for _, code in EXIT_CASES}
    assert len(codes) == len(EXIT_CASES)
