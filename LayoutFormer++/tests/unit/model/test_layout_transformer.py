import torch

from layoutformer_pp.model.layout_transformer.model import (
    LayoutTransformer,
    generate_square_subsequent_mask,
    top_k_logits,
)


def test_generate_square_subsequent_mask_shape():
    mask = generate_square_subsequent_mask(4)
    assert mask.shape == (4, 4)
    assert torch.isinf(mask).any()


def test_top_k_logits_masks_values():
    logits = torch.tensor([[1.0, 0.5, -1.0]])
    out = top_k_logits(logits, k=1)
    assert torch.isinf(out[0, 1])
    assert torch.isinf(out[0, 2])
    assert out[0, 0].item() == 1.0


def test_layout_transformer_forward_and_generate():
    model = LayoutTransformer(
        vocab_size=12,
        max_len=16,
        bos_token_id=1,
        pad_token_id=2,
        eos_token_id=3,
        d_model=16,
        nhead=4,
        num_layers=2,
        dropout=0.1,
        share_embedding=False,
    )
    input_ids = torch.randint(0, 12, (2, 5))
    padding_mask = torch.zeros(2, 5, dtype=torch.bool)
    labels = torch.randint(0, 12, (2, 5))

    outputs = model(input_ids, padding_mask, labels)
    assert "loss" in outputs
    assert outputs["logits"].shape == (2, 5, 12)

    generated = model(input_ids, padding_mask, max_length=6)
    assert generated["output"].shape[0] == 2
