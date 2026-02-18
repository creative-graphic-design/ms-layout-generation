from __future__ import annotations

import json
from types import SimpleNamespace
from pathlib import Path

import torch

from improved_diffusion import rounding
from improved_diffusion import test_util


def _prepare_vocab(tmp_path: Path) -> Path:
    vocab = {chr(ord("A") + idx): idx for idx in range(8)}
    vocab_path = tmp_path / "vocab.json"
    vocab_path.write_text(json.dumps(vocab), encoding="utf-8")
    return vocab_path


def _prepare_embedding(tmp_path: Path, emb_dim: int = 4) -> Path:
    model = torch.nn.Embedding(8, emb_dim)
    path = tmp_path / "random_emb.torch"
    torch.save(model.state_dict(), path)
    return path


def test_load_models_and_rounding(tmp_path: Path) -> None:
    _prepare_vocab(tmp_path)
    _prepare_embedding(tmp_path)

    model, tokenizer = rounding.load_models(
        modality="e2e",
        mode="random",
        model_name_or_path=str(tmp_path),
        emb_dim=4,
        file=str(tmp_path),
    )
    assert model.weight.shape[1] == 4
    assert isinstance(tokenizer, dict)

    embeddings = [torch.zeros(8, 4)]
    decoded = rounding.rounding_func("random", embeddings, model, tokenizer)
    assert len(decoded) == 1


def test_load_tokenizer(tmp_path: Path) -> None:
    _prepare_vocab(tmp_path)
    tokenizer = rounding.load_tokenizer("e2e", "random", str(tmp_path))
    assert isinstance(tokenizer, dict)


def test_test_util_helpers() -> None:
    model = torch.nn.Embedding(3, 4)
    args = SimpleNamespace(model_arch="transformer", emb_scale_factor=1.0, ungen=False)

    input_ids = torch.tensor([[0, 1]])
    x = torch.zeros(1, 2, 4)
    loss = test_util.compute_logp(args, model, x, input_ids)
    assert loss.shape == (1, 2)

    weights = test_util.get_weights(model, args)
    assert weights.weight.requires_grad is False

    rounded = test_util.denoised_fn_round(args, weights, x, t=None)
    assert rounded.shape == x.shape


def test_load_results(tmp_path: Path) -> None:
    path = tmp_path / "out.json"
    test_util.load_results(str(path), {"a": 1})
    assert path.exists()
