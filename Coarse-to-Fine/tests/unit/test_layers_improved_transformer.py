"""Unit tests for c2f_model.layers.improved_transformer."""

import torch

from coarse_to_fine.c2f_model.layers.improved_transformer import (
    TransformerEncoderLayerImproved,
    TransformerDecoderLayerImproved,
    TransformerDecoderLayerGlobalImproved,
)


def test_encoder_layer_improved():
    layer = TransformerEncoderLayerImproved(d_model=8, nhead=2)
    src = torch.randn(4, 2, 8)
    out = layer(src)
    assert out.shape == (4, 2, 8)


def test_encoder_layer_improved_with_memory2():
    layer = TransformerEncoderLayerImproved(d_model=8, nhead=2, d_global2=8)
    src = torch.randn(4, 2, 8)
    memory2 = torch.randn(4, 2, 8)
    out = layer(src, memory2=memory2)
    assert out.shape == (4, 2, 8)


def test_decoder_layer_improved():
    layer = TransformerDecoderLayerImproved(d_model=8, nhead=2)
    tgt = torch.randn(3, 2, 8)
    memory = torch.randn(4, 2, 8)
    out = layer(tgt, memory)
    assert out.shape == (3, 2, 8)


def test_decoder_layer_global_improved():
    layer = TransformerDecoderLayerGlobalImproved(d_model=8, d_global=8, nhead=2)
    tgt = torch.randn(3, 2, 8)
    memory = torch.randn(2, 8)
    out = layer(tgt, memory)
    assert out.shape == (3, 2, 8)
