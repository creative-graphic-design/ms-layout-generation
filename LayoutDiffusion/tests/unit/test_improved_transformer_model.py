from __future__ import annotations

import torch
from transformers import BertConfig

from improved_diffusion.transformer_model import (
    TransformerModel,
    DiscreteTransformerModel,
)


def _tiny_config() -> BertConfig:
    return BertConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        max_position_embeddings=16,
    )


def test_transformer_model_forward() -> None:
    config = _tiny_config()
    model = TransformerModel(
        in_channels=4,
        model_channels=8,
        out_channels=4,
        num_res_blocks=1,
        dropout=0.0,
        num_heads=2,
        use_checkpoint=False,
        training_mode="e2e",
        vocab_size=10,
        experiment_mode="lm",
        config=config,
    )

    x = torch.randn(2, 4, 4)
    timesteps = torch.tensor([1, 2])
    out = model(x, timesteps)
    assert out.shape == x.shape


def test_discrete_transformer_model_forward() -> None:
    config = _tiny_config()
    model = DiscreteTransformerModel(
        in_channels=4,
        model_channels=8,
        num_res_blocks=1,
        dropout=0.0,
        training_mode="discrete",
        vocab_size=159,
        config=config,
    )

    x = torch.randint(0, 10, (2, 4))
    timesteps = torch.tensor([0, 1])
    out = model(x, timesteps)
    assert out.shape[0] == 2
    assert out.shape[2] == 4
