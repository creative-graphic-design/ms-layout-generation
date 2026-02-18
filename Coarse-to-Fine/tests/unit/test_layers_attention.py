"""Unit tests for c2f_model.layers.attention.MultiheadAttention."""

import torch

from coarse_to_fine.c2f_model.layers.attention import MultiheadAttention


def test_multihead_attention_self():
    attn = MultiheadAttention(embed_dim=8, num_heads=2)
    query = torch.randn(5, 2, 8)
    out, weights = attn(query, query, query)
    assert out.shape == (5, 2, 8)
    assert weights.shape == (2, 5, 5)


def test_multihead_attention_cross_with_kv_dim():
    attn = MultiheadAttention(embed_dim=8, num_heads=2, kdim=4, vdim=4)
    query = torch.randn(5, 2, 8)
    key = torch.randn(6, 2, 4)
    value = torch.randn(6, 2, 4)
    attn_mask = torch.zeros(5, 6)
    key_padding_mask = torch.zeros(2, 6, dtype=torch.bool)
    out, weights = attn(query, key, value, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
    assert out.shape == (5, 2, 8)
    assert weights.shape == (2, 5, 6)


def test_multihead_attention_bias_false():
    attn = MultiheadAttention(embed_dim=8, num_heads=2, bias=False)
    query = torch.randn(3, 2, 8)
    out, weights = attn(query, query, query)
    assert out.shape == (3, 2, 8)
    assert weights.shape == (2, 3, 3)


def test_multihead_attention_add_bias_kv_and_zero_attn():
    attn = MultiheadAttention(embed_dim=8, num_heads=2, add_bias_kv=True, add_zero_attn=True)
    query = torch.randn(3, 2, 8)
    key = torch.randn(3, 2, 8)
    value = torch.randn(3, 2, 8)
    out, weights = attn(query, key, value)
    assert out.shape == (3, 2, 8)
    assert weights.shape[0] == 2


def test_multihead_attention_setstate():
    attn = MultiheadAttention(embed_dim=8, num_heads=2)
    state = {}
    attn.__setstate__(state)
    assert attn._qkv_same_embed_dim is True
