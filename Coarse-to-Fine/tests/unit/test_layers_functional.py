"""Unit tests for c2f_model.layers.functional.multi_head_attention_forward."""

import pytest
import torch

from coarse_to_fine.c2f_model.layers.functional import multi_head_attention_forward


def _base_params(embed_dim):
    in_proj_weight = torch.randn(3 * embed_dim, embed_dim)
    in_proj_bias = torch.randn(3 * embed_dim)
    out_proj_weight = torch.randn(embed_dim, embed_dim)
    out_proj_bias = torch.randn(embed_dim)
    return in_proj_weight, in_proj_bias, out_proj_weight, out_proj_bias


def test_multi_head_attention_forward_self():
    embed_dim = 8
    num_heads = 2
    query = torch.randn(4, 2, embed_dim)
    in_proj_weight, in_proj_bias, out_proj_weight, out_proj_bias = _base_params(embed_dim)

    out, weights = multi_head_attention_forward(
        query,
        query,
        query,
        embed_dim,
        num_heads,
        in_proj_weight,
        in_proj_bias,
        None,
        None,
        False,
        0.0,
        out_proj_weight,
        out_proj_bias,
        training=True,
        key_padding_mask=None,
        need_weights=True,
        attn_mask=None,
    )

    assert out.shape == (4, 2, embed_dim)
    assert weights.shape == (2, 4, 4)


def test_multi_head_attention_forward_cross():
    embed_dim = 8
    num_heads = 2
    query = torch.randn(3, 2, embed_dim)
    key = torch.randn(5, 2, embed_dim)
    value = torch.randn(5, 2, embed_dim)
    in_proj_weight, in_proj_bias, out_proj_weight, out_proj_bias = _base_params(embed_dim)

    attn_mask = torch.zeros(3, 5)
    out, weights = multi_head_attention_forward(
        query,
        key,
        value,
        embed_dim,
        num_heads,
        in_proj_weight,
        in_proj_bias,
        None,
        None,
        False,
        0.0,
        out_proj_weight,
        out_proj_bias,
        training=True,
        key_padding_mask=None,
        need_weights=True,
        attn_mask=attn_mask,
    )

    assert out.shape == (3, 2, embed_dim)
    assert weights.shape == (2, 3, 5)


def test_multi_head_attention_forward_separate_proj():
    embed_dim = 8
    num_heads = 2
    query = torch.randn(3, 2, embed_dim)
    key = torch.randn(3, 2, embed_dim)
    value = torch.randn(3, 2, embed_dim)

    q_proj_weight = torch.randn(embed_dim, embed_dim)
    k_proj_weight = torch.randn(embed_dim, embed_dim)
    v_proj_weight = torch.randn(embed_dim, embed_dim)
    in_proj_bias = torch.randn(3 * embed_dim)
    out_proj_weight = torch.randn(embed_dim, embed_dim)
    out_proj_bias = torch.randn(embed_dim)

    out, weights = multi_head_attention_forward(
        query,
        key,
        value,
        embed_dim,
        num_heads,
        None,
        in_proj_bias,
        None,
        None,
        False,
        0.0,
        out_proj_weight,
        out_proj_bias,
        training=True,
        key_padding_mask=None,
        need_weights=True,
        attn_mask=None,
        use_separate_proj_weight=True,
        q_proj_weight=q_proj_weight,
        k_proj_weight=k_proj_weight,
        v_proj_weight=v_proj_weight,
    )

    assert out.shape == (3, 2, embed_dim)
    assert weights.shape == (2, 3, 3)


def test_multi_head_attention_forward_invalid_mask():
    embed_dim = 8
    num_heads = 2
    query = torch.randn(3, 2, embed_dim)
    in_proj_weight, in_proj_bias, out_proj_weight, out_proj_bias = _base_params(embed_dim)

    attn_mask = torch.zeros(1, 2, 3, 4)
    with pytest.raises(RuntimeError):
        multi_head_attention_forward(
            query,
            query,
            query,
            embed_dim,
            num_heads,
            in_proj_weight,
            in_proj_bias,
            None,
            None,
            False,
            0.0,
            out_proj_weight,
            out_proj_bias,
            training=True,
            key_padding_mask=None,
            need_weights=True,
            attn_mask=attn_mask,
        )


def test_multi_head_attention_forward_bias_and_zero_attn():
    embed_dim = 8
    num_heads = 2
    query = torch.randn(3, 2, embed_dim)
    key = torch.randn(3, 2, embed_dim)
    value = torch.randn(3, 2, embed_dim)
    in_proj_weight, in_proj_bias, out_proj_weight, out_proj_bias = _base_params(embed_dim)
    bias_k = torch.randn(1, 1, embed_dim)
    bias_v = torch.randn(1, 1, embed_dim)
    key_padding_mask = torch.zeros(2, 3, dtype=torch.bool)

    out, weights = multi_head_attention_forward(
        query,
        key,
        value,
        embed_dim,
        num_heads,
        in_proj_weight,
        in_proj_bias,
        bias_k,
        bias_v,
        True,
        0.0,
        out_proj_weight,
        out_proj_bias,
        training=True,
        key_padding_mask=key_padding_mask,
        need_weights=True,
        attn_mask=None,
    )

    assert out.shape == (3, 2, embed_dim)
    assert weights.shape[0] == 2
