"""Unit tests for basic_blocks layers."""

import torch

from coarse_to_fine.c2f_model.layers.basic_blocks import FCN, HierarchFCN, ResNet


def test_fcn_forward():
    model = FCN(d_model=8, n_commands=4, n_args=2, args_dim=6)
    x = torch.randn(3, 2, 8)
    command_logits, args_logits = model(x)
    assert command_logits.shape == (3, 2, 4)
    assert args_logits.shape == (3, 2, 2, 6)


def test_hierarch_fcn_forward():
    model = HierarchFCN(d_model=8, dim_z=4)
    x = torch.randn(3, 2, 8)
    visibility_logits, z = model(x)
    assert visibility_logits.shape == (1, 3, 2, 2)
    assert z.shape == (1, 3, 2, 4)


def test_resnet_forward():
    model = ResNet(d_model=8)
    x = torch.randn(2, 8)
    out = model(x)
    assert out.shape == x.shape
