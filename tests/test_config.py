"""Layered config precedence: CLI > env > file(model>cap>global) > builtin."""

from __future__ import annotations

from localai.capabilities.image.text_to_image.models import SCHNELL
from localai.core.config import load_settings

CAP = "text-to-image"
MODEL = "schnell"


def test_builtin_model_default(monkeypatch):
    monkeypatch.delenv("LOCALAI_STEPS", raising=False)
    settings = load_settings(CAP, MODEL, spec=SCHNELL)
    assert settings.get_int("steps") == SCHNELL.default_steps  # 4


def test_file_layer_precedence(tmp_path, monkeypatch):
    monkeypatch.delenv("LOCALAI_STEPS", raising=False)
    cfg = tmp_path / "localai.toml"

    cfg.write_text("[defaults]\nsteps = 1\n", encoding="utf-8")
    assert load_settings(CAP, MODEL, config_path=str(cfg), spec=SCHNELL).get_int("steps") == 1

    cfg.write_text("[defaults]\nsteps = 1\n[text-to-image]\nsteps = 2\n", encoding="utf-8")
    assert load_settings(CAP, MODEL, config_path=str(cfg), spec=SCHNELL).get_int("steps") == 2

    cfg.write_text(
        "[defaults]\nsteps = 1\n[text-to-image]\nsteps = 2\n"
        "[text-to-image.models.schnell]\nsteps = 3\n",
        encoding="utf-8",
    )
    assert load_settings(CAP, MODEL, config_path=str(cfg), spec=SCHNELL).get_int("steps") == 3


def test_full_precedence_ladder(tmp_path, monkeypatch):
    cfg = tmp_path / "localai.toml"
    cfg.write_text(
        "[defaults]\nsteps = 1\n[text-to-image]\nsteps = 2\n"
        "[text-to-image.models.schnell]\nsteps = 3\n",
        encoding="utf-8",
    )

    # file model layer wins over capability/global/builtin
    monkeypatch.delenv("LOCALAI_STEPS", raising=False)
    assert load_settings(CAP, MODEL, config_path=str(cfg), spec=SCHNELL).get_int("steps") == 3

    # env beats the file
    monkeypatch.setenv("LOCALAI_STEPS", "7")
    assert load_settings(CAP, MODEL, config_path=str(cfg), spec=SCHNELL).get_int("steps") == 7

    # CLI beats env
    out = load_settings(
        CAP, MODEL, cli_overrides={"steps": 9}, config_path=str(cfg), spec=SCHNELL
    )
    assert out.get_int("steps") == 9


def test_env_mapping_typed(monkeypatch):
    monkeypatch.setenv("LOCALAI_OUTPUT_DIR", "myout")
    monkeypatch.setenv("LOCALAI_BATCH", "3")
    settings = load_settings(CAP, MODEL, spec=SCHNELL)
    assert str(settings.output_dir) == "myout"
    assert settings.batch == 3
