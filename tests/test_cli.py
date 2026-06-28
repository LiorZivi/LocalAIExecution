"""Top-level CLI dispatch + the shared --json contract, driven by the dummy."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from localai.core import cli


def test_no_command_prints_help(capsys):
    assert cli.main([]) == 0
    assert "usage" in capsys.readouterr().out.lower()


def test_capabilities_listing(dummy_adapter, capsys):
    assert cli.main(["capabilities"]) == 0
    out = capsys.readouterr().out
    assert "dummy" in out
    assert "text-to-image" in out


def test_capabilities_json(dummy_adapter, capsys):
    assert cli.main(["capabilities", "--json"]) == 0
    data = json.loads(capsys.readouterr().out.strip())
    ids = {c["id"] for c in data["capabilities"]}
    assert "dummy" in ids and "text-to-image" in ids


def test_dummy_run_json_contract(dummy_adapter, tmp_path, capsys):
    code = cli.main(["dummy-run", "hello world", "--json", "--output-dir", str(tmp_path)])
    assert code == 0
    out = capsys.readouterr().out.strip()
    data = json.loads(out)  # exactly one JSON object on stdout
    assert data["capability"] == "dummy"
    assert data["model"] == "m1"
    artifact = data["artifacts"][0]
    assert artifact["type"] == "dummytext"
    assert Path(artifact["path"]).exists()
    assert "metadata" in artifact


def test_dummy_run_human_prints_path(dummy_adapter, tmp_path, capsys):
    code = cli.main(["dummy-run", "hi", "--output-dir", str(tmp_path)])
    assert code == 0
    out = capsys.readouterr().out.strip().splitlines()
    assert out and Path(out[-1]).exists()  # final stdout line is a saved path


def test_unknown_subcommand_exits_2(dummy_adapter):
    with pytest.raises(SystemExit) as exc:
        cli.main(["definitely-not-a-command"])
    assert exc.value.code == 2
