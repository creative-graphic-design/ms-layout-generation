"""Unit tests for positional encoding layers."""

import torch

from coarse_to_fine.c2f_model.layers.positional_encoding import (
    PositionalEncodingSinCos,
    PositionalEncodingLUT,
)


def test_positional_encoding_sincos():
    layer = PositionalEncodingSinCos(d_model=8, dropout=0.0, max_len=16)
    x = torch.zeros(10, 2, 8)
    out = layer(x)
    assert out.shape == x.shape


def test_positional_encoding_lut():
    layer = PositionalEncodingLUT(d_model=8, dropout=0.0, max_len=16)
    x = torch.zeros(10, 2, 8)
    out = layer(x)
    assert out.shape == x.shape
