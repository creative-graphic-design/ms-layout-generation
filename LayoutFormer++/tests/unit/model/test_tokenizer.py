from layoutformer_pp.model.layout_transformer.tokenizer import (
    LayoutTransformerTokenizer,
)


def test_tokenizer_padding_and_masks():
    tokenizer = LayoutTransformerTokenizer(tokens=["label_1", "0", "1"])
    batch = tokenizer(["label_1 0", "label_1 0 1"], add_eos=True, add_bos=False)

    assert batch["input_ids"].shape == (2, 4)
    assert batch["mask"].shape == (2, 4)
    assert batch["mask"][0, -1].item() is False
    assert batch["input_ids"][0, -1].item() == tokenizer.pad_token_id


def test_tokenizer_decode_roundtrip():
    tokenizer = LayoutTransformerTokenizer(tokens=["label_1", "0", "1"])
    encoded = tokenizer("label_1 0", add_eos=True, add_bos=False)
    decoded = tokenizer.decode(
        encoded["input_ids"][0].tolist(), skip_special_tokens=True
    )

    assert "label_1" in decoded
    assert "0" in decoded
    assert "<eos>" not in decoded
