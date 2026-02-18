import torch

from layoutformer_pp.model.layout_transformer.constrained_decoding import (
    TransformerSortByDictLabelConstraint,
    TransformerSortByDictLabelSizeConstraint,
    TransformerSortByDictRelationConstraint,
)
from layoutformer_pp.model.layout_transformer.tokenizer import (
    LayoutTransformerTokenizer,
)
from layoutformer_pp.tasks.gen_r import RelationTypes


def _make_tokenizer():
    tokens = [
        "text",
        "image",
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "|",
    ]
    return LayoutTransformerTokenizer(tokens)


def test_label_constraint_progression():
    tokenizer = _make_tokenizer()
    index2label = {1: "Text", 2: "Image"}
    constraint = TransformerSortByDictLabelConstraint(
        tokenizer,
        discrete_degree=4,
        label_set=["Text", "Image"],
        index2label=index2label,
        add_sep_token=False,
    )
    constraint.prepare([torch.tensor([1, 2])])

    text_id = tokenizer.convert_tokens_to_ids(["text"])[0]
    image_id = tokenizer.convert_tokens_to_ids(["image"])[0]
    num_ids = set(tokenizer.convert_tokens_to_ids(["0", "1", "2", "3"]))

    p0, _ = constraint(0, 0, torch.tensor([], dtype=torch.long))
    assert set(p0) == {text_id}

    p1, _ = constraint(0, 1, torch.tensor([], dtype=torch.long))
    assert set(p1) == num_ids
    constraint(0, 2, torch.tensor([], dtype=torch.long))
    constraint(0, 3, torch.tensor([], dtype=torch.long))
    constraint(0, 4, torch.tensor([], dtype=torch.long))
    p5, _ = constraint(0, 5, torch.tensor([], dtype=torch.long))
    assert set(p5) == {image_id}


def test_label_size_constraint_enforces_sizes_and_sep():
    tokenizer = _make_tokenizer()
    index2label = {1: "Text"}
    constraint = TransformerSortByDictLabelSizeConstraint(
        tokenizer,
        discrete_degree=6,
        label_set=["Text"],
        index2label=index2label,
        add_sep_token=True,
        sep_token="|",
    )
    constraint.prepare([torch.tensor([1])], [torch.tensor([[1, 2, 3, 4]])])

    width_id = tokenizer.convert_tokens_to_ids(["3"])[0]
    height_id = tokenizer.convert_tokens_to_ids(["4"])[0]
    special_ids = {tokenizer.pad_token_id, tokenizer.eos_token_id}

    constraint(0, 0, torch.tensor([], dtype=torch.long))
    constraint(0, 1, torch.tensor([], dtype=torch.long))
    constraint(0, 2, torch.tensor([], dtype=torch.long))
    p3, _ = constraint(0, 3, torch.tensor([], dtype=torch.long))
    assert set(p3) == {width_id}

    p4, _ = constraint(0, 4, torch.tensor([], dtype=torch.long))
    assert set(p4) == {height_id}

    p5, _ = constraint(0, 5, torch.tensor([], dtype=torch.long))
    assert set(p5) == special_ids


def test_relation_constraint_allows_numbers_without_relations():
    tokenizer = _make_tokenizer()
    index2label = {1: "Text"}
    rel_index2type = RelationTypes.index2type()
    constraint = TransformerSortByDictRelationConstraint(
        tokenizer,
        discrete_degree=4,
        label_set=["Text"],
        index2label=index2label,
        rel_index2type=rel_index2type,
    )
    constraint.prepare([torch.tensor([1])], [list()])

    num_ids = set(tokenizer.convert_tokens_to_ids(["0", "1", "2", "3"]))

    constraint(0, 0, torch.tensor([], dtype=torch.long))
    p1, _ = constraint(0, 1, torch.tensor([], dtype=torch.long))
    assert set(p1) == num_ids
