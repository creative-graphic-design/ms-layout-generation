from __future__ import annotations

import torch

from improved_diffusion.discrete_diffusion import (
    DiffusionTransformer,
    gaussian_matrix2,
    index_to_log_onehot,
    log_add_exp,
    log_onehot_to_index,
    log_1_min_a,
)


class DummyDiscreteModel(torch.nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, x, t, y=None):
        bsz, seq_len = x.shape
        return torch.zeros(bsz, self.num_classes - 1, seq_len)


def test_basic_log_helpers() -> None:
    a = torch.tensor([0.0])
    b = torch.tensor([-1.0])
    out = log_add_exp(a, b)
    assert out.numel() == 1

    log_x = torch.log(torch.tensor([0.5]))
    assert torch.isfinite(log_1_min_a(log_x))

    mat = gaussian_matrix2(0, bt=torch.tensor([0.1, 0.2]))
    assert mat.shape == (128, 128)


def test_diffusion_transformer_core() -> None:
    diffusion = DiffusionTransformer(
        diffusion_step=10,
        alpha_init_type="gaussian_refine_pow2.5",
        num_classes=159,
        content_seq_len=4,
        matrix_policy=1,
    )

    x_start_idx = torch.randint(0, 158, (2, 4))
    log_x_start = index_to_log_onehot(x_start_idx, diffusion.num_classes)
    t = torch.tensor([0, 1])

    log_qt = diffusion.q_pred(log_x_start, t)
    assert log_qt.shape[-1] == 4

    log_qt_one = diffusion.q_pred_one_timestep(log_x_start, t)
    assert log_qt_one.shape[-1] == 4

    log_post = diffusion.q_posterior(log_x_start, log_x_start, t)
    assert log_post.shape[-1] == 4

    sample = diffusion.q_sample(log_x_start, t)
    idx = log_onehot_to_index(sample)
    assert idx.shape == (2, 4)

    sample_one = diffusion.q_sample_onestep(log_x_start, t)
    idx_one = log_onehot_to_index(sample_one)
    assert idx_one.shape == (2, 4)

    model = DummyDiscreteModel(diffusion.num_classes)
    log_pred = diffusion.predict_start(log_x_start, model, y=x_start_idx, t=t)
    assert log_pred.shape[1] == diffusion.num_classes

    out = diffusion.p_sample(log_x_start, model, t)
    assert out.shape[1] == diffusion.num_classes

    t_sample, pt = diffusion.sample_time(2, device=log_x_start.device, method="uniform")
    assert t_sample.shape == (2,)
    assert pt.shape == (2,)
