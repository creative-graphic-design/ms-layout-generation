"""Unit tests for c2f_model.layers.utils helpers."""

import torch

from coarse_to_fine.c2f_model.layers import utils


def test_make_seq_first_and_batch_first():
    x = torch.randn(2, 3, 4)
    seq = utils._make_seq_first(x)
    assert seq.shape == (3, 2, 4)
    back = utils._make_batch_first(seq)
    assert back.shape == x.shape

    y = torch.randn(2, 3, 4)
    seq_x, seq_y = utils._make_seq_first(x, y)
    assert seq_x.shape == (3, 2, 4)
    assert seq_y.shape == (3, 2, 4)
    back_x, back_y = utils._make_batch_first(seq_x, seq_y)
    assert back_x.shape == x.shape
    assert back_y.shape == y.shape


def test_key_padding_and_padding_mask():
    mask = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.bool)
    key_padding = utils._get_key_padding_mask(mask)
    assert key_padding.shape == mask.shape

    padding_mask = utils._get_padding_mask(mask)
    assert padding_mask.shape == (3, 2, 1)


def test_generate_square_subsequent_mask():
    mask = utils._generate_square_subsequent_mask(4)
    assert mask.shape == (4, 4)
    assert torch.isfinite(mask[0, 0])


def test_pack_and_unpack_group_batch():
    x = torch.randn(2, 3, 4, 5)
    packed = utils._pack_group_batch(x)
    assert packed.shape == (2, 12, 5)
    unpacked = utils._unpack_group_batch(4, packed)
    assert unpacked.shape == x.shape

    y = torch.randn(2, 3, 4, 6)
    packed_x, packed_y = utils._pack_group_batch(x, y)
    assert packed_x.shape == (2, 12, 5)
    assert packed_y.shape == (2, 12, 6)
    unpacked_x, unpacked_y = utils._unpack_group_batch(4, packed_x, packed_y)
    assert unpacked_x.shape == x.shape
    assert unpacked_y.shape == y.shape


def test_make_group_first():
    x = torch.randn(2, 3, 4, 5)
    grouped = utils._make_group_first(x)
    assert grouped.shape == (3, 2, 4, 5)
