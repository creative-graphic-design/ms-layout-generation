from __future__ import annotations

from pathlib import Path

from transformers import BertConfig

from improved_diffusion import script_util


def _write_config(tmp_path: Path) -> str:
    cfg = BertConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        max_position_embeddings=32,
    )
    config_dir = tmp_path / "bert"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "config.json").write_text(cfg.to_json_string(), encoding="utf-8")
    return str(config_dir)


def test_str2bool() -> None:
    assert script_util.str2bool("yes") is True
    assert script_util.str2bool("no") is False


def test_args_to_dict() -> None:
    class Obj:
        a = 1
        b = 2

    out = script_util.args_to_dict(Obj, ["a", "b"])
    assert out == {"a": 1, "b": 2}


def test_create_model_and_diffusion(tmp_path: Path) -> None:
    config_name = _write_config(tmp_path)
    defaults = script_util.model_and_diffusion_defaults()
    defaults.update(
        {
            "diffusion_steps": 10,
            "seq_length": 4,
            "num_channels": 8,
            "num_res_blocks": 1,
            "model_arch": "transformer",
            "training_mode": "discrete",
            "noise_schedule": "gaussian_refine_pow2.5",
            "vocab_size": 159,
            "config_name": config_name,
            "dropout": 0.0,
        }
    )
    model, diffusion = script_util.create_model_and_diffusion(**defaults)
    assert model is not None
    assert diffusion is not None
