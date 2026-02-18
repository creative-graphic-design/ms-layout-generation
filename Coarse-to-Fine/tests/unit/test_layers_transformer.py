"""Unit tests for c2f_model.layers.transformer."""

import pytest
import torch

from coarse_to_fine.c2f_model.layers.transformer import (
    Transformer,
    TransformerEncoderLayer,
    TransformerDecoderLayer,
    _get_activation_fn,
)


def test_transformer_forward():
    model = Transformer(d_model=8, nhead=2, num_encoder_layers=1, num_decoder_layers=1)
    src = torch.randn(5, 2, 8)
    tgt = torch.randn(3, 2, 8)
    out = model(src, tgt)
    assert out.shape == (3, 2, 8)


def test_encoder_decoder_layers():
    enc = TransformerEncoderLayer(d_model=8, nhead=2)
    dec = TransformerDecoderLayer(d_model=8, nhead=2)
    src = torch.randn(4, 2, 8)
    tgt = torch.randn(3, 2, 8)
    memory = enc(src)
    out = dec(tgt, memory)
    assert out.shape == (3, 2, 8)


def test_get_activation_fn():
    assert _get_activation_fn("relu") is not None
    assert _get_activation_fn("gelu") is not None
    with pytest.raises(RuntimeError):
        _get_activation_fn("tanh")
