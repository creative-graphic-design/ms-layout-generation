from __future__ import annotations

import torch

from improved_diffusion import fp16_util, losses, nn


def test_losses_outputs() -> None:
    mean1 = torch.zeros(2, 3)
    logvar1 = torch.zeros(2, 3)
    mean2 = torch.ones(2, 3)
    logvar2 = torch.zeros(2, 3)
    kl = losses.normal_kl(mean1, logvar1, mean2, logvar2)
    assert kl.shape == (2, 3)

    cdf = losses.approx_standard_normal_cdf(torch.tensor([0.0]))
    assert 0.4 < cdf.item() < 0.6

    x = torch.zeros(1, 2)
    means = torch.zeros_like(x)
    log_scales = torch.zeros_like(x)
    ll = losses.discretized_gaussian_log_likelihood(
        x, means=means, log_scales=log_scales
    )
    assert ll.shape == x.shape

    ll_text = losses.discretized_text_log_likelihood(
        x, means=means, log_scales=log_scales
    )
    assert ll_text.shape == x.shape

    logp = losses.gaussian_density(x, means=means, log_scales=log_scales)
    assert logp.shape == x.shape


def test_nn_helpers() -> None:
    conv = nn.conv_nd(1, 1, 2, 3)
    assert isinstance(conv, torch.nn.Conv1d)

    pool = nn.avg_pool_nd(2, 2)
    assert isinstance(pool, torch.nn.AvgPool2d)

    emb = nn.timestep_embedding(torch.tensor([1, 2]), dim=6)
    assert emb.shape == (2, 6)

    module = torch.nn.Linear(2, 2)
    nn.zero_module(module)
    for p in module.parameters():
        assert torch.allclose(p, torch.zeros_like(p))

    module = torch.nn.Linear(2, 2)
    nn.scale_module(module, 0.5)

    src = [torch.nn.Parameter(torch.ones(2, 2))]
    tgt = [torch.nn.Parameter(torch.zeros(2, 2))]
    nn.update_ema(tgt, src, rate=0.5)
    assert torch.allclose(tgt[0], torch.full((2, 2), 0.5))


def test_fp16_utils() -> None:
    conv = torch.nn.Conv2d(1, 1, 3)
    fp16_util.convert_module_to_f16(conv)
    assert conv.weight.dtype == torch.float16

    fp16_util.convert_module_to_f32(conv)
    assert conv.weight.dtype == torch.float32

    model = torch.nn.Linear(2, 2)
    master = fp16_util.make_master_params(model.parameters())
    assert len(master) == 1

    for p in model.parameters():
        p.grad = torch.ones_like(p)
    fp16_util.model_grads_to_master_grads(model.parameters(), master)
    assert master[0].grad is not None

    fp16_util.master_params_to_model_params(model.parameters(), master)

    fp16_util.zero_grad(model.parameters())
    for p in model.parameters():
        assert p.grad is None or torch.all(p.grad == 0)
