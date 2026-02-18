import torch

from parse_then_place.semantic_parser.dataset.dataset import CollateFn


def test_collate_fn_pads_to_max_lengths():
    examples = [
        {
            "text_ids": torch.tensor([1, 2]),
            "text_attention_mask": torch.tensor([1, 1]),
            "lf_ids": torch.tensor([5]),
            "lf_attention_mask": torch.tensor([1]),
        },
        {
            "text_ids": torch.tensor([3, 4, 5]),
            "text_attention_mask": torch.tensor([1, 1, 1]),
            "lf_ids": torch.tensor([6, 7]),
            "lf_attention_mask": torch.tensor([1, 1]),
        },
    ]
    collate = CollateFn(pad_id=0)
    batch = collate(examples)

    assert batch["text_ids"].shape == (2, 3)
    assert batch["text_attention_mask"].shape == (2, 3)
    assert batch["lf_ids"].shape == (2, 2)
    assert batch["lf_attention_mask"].shape == (2, 2)

    assert batch["text_ids"][0, -1].item() == 0
    assert batch["text_attention_mask"][0, -1].item() == 0
    assert batch["lf_ids"][0, -1].item() == 0
    assert batch["lf_attention_mask"][0, -1].item() == 0
