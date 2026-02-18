import torch
import pytest

from layoutformer_pp.model.layout_transformer.tokenizer import (
    LayoutTransformerTokenizer,
)
import layoutformer_pp.model.utils as model_utils
from layoutformer_pp.model.layout_transformer import constrained_decoding


def _make_tokenizer():
    tokens = ["label_1", "label_2", "|"] + [str(i) for i in range(10)]
    return LayoutTransformerTokenizer(tokens)


def test_label_constraint_updates_decode_state():
    tokenizer = _make_tokenizer()
    index2label = {1: "label_1", 2: "label_2"}
    constraint = model_utils.LabelConstraint(
        tokenizer,
        discrete_degree=5,
        label_set=["label_1", "label_2"],
        index2label=index2label,
    )
    constraint.prepare([torch.tensor([1, 2])])

    label_1_id = tokenizer.convert_tokens_to_ids(["label_1"])[0]
    allow = constraint(0, torch.tensor([label_1_id]))
    assert tokenizer.eos_token_id not in allow

    token_ids = torch.tensor(tokenizer.convert_tokens_to_ids(["label_1", "label_2"]))
    allow = constraint(0, token_ids)
    assert tokenizer.eos_token_id in allow


def test_label_constraint_sep_token_updates_state():
    tokenizer = _make_tokenizer()
    index2label = {1: "label_1"}
    constraint = model_utils.LabelConstraint(
        tokenizer,
        discrete_degree=3,
        label_set=["label_1"],
        index2label=index2label,
        add_sep_token=True,
        sep_token="|",
    )
    constraint.prepare([torch.tensor([1])])
    sep_id = tokenizer.convert_tokens_to_ids(["|"])[0]
    allow = constraint(0, torch.tensor([sep_id]))
    assert sep_id in allow


@pytest.mark.parametrize("module", [model_utils, constrained_decoding])
def test_transformer_sort_by_dict_label_constraint_sequence(module):
    tokenizer = _make_tokenizer()
    index2label = {1: "label_1", 2: "label_2"}
    constraint = module.TransformerSortByDictLabelConstraint(
        tokenizer,
        discrete_degree=5,
        label_set=["label_1", "label_2"],
        index2label=index2label,
        add_sep_token=True,
        sep_token="|",
    )
    constraint.prepare([torch.tensor([1, 2])])

    label_id = tokenizer.convert_tokens_to_ids(["label_1"])[0]
    num_id = tokenizer.convert_tokens_to_ids(["1"])[0]

    out, _ = constraint(0, 0, torch.tensor([label_id]))
    assert out

    for seq_id in range(1, 5):
        out, _ = constraint(0, seq_id, torch.tensor([num_id]))

    sep_id = tokenizer.convert_tokens_to_ids(["|"])[0]
    out, _ = constraint(0, 5, torch.tensor([sep_id]))
    assert sep_id in out


@pytest.mark.parametrize("module", [model_utils, constrained_decoding])
def test_transformer_sort_by_dict_label_size_constraint_sequence(module):
    tokenizer = _make_tokenizer()
    index2label = {1: "label_1", 2: "label_2"}
    constraint = module.TransformerSortByDictLabelSizeConstraint(
        tokenizer,
        discrete_degree=5,
        label_set=["label_1", "label_2"],
        index2label=index2label,
    )
    label_ids = [torch.tensor([1, 2])]
    bboxes = [torch.tensor([[0, 0, 2, 3], [1, 1, 3, 4]])]
    constraint.prepare(label_ids, bboxes)

    label_id = tokenizer.convert_tokens_to_ids(["label_1"])[0]
    num_id = tokenizer.convert_tokens_to_ids(["1"])[0]
    size_id = tokenizer.convert_tokens_to_ids(["2"])[0]

    out, _ = constraint(0, 0, torch.tensor([label_id]))
    assert out

    out, _ = constraint(0, 1, torch.tensor([num_id]))
    out, _ = constraint(0, 2, torch.tensor([num_id]))
    out, _ = constraint(0, 3, torch.tensor([size_id]))
    out, _ = constraint(0, 4, torch.tensor([size_id]))
    assert out


@pytest.mark.parametrize("module", [model_utils, constrained_decoding])
def test_transformer_sort_by_dict_relation_constraint_sequence(module):
    tokenizer = _make_tokenizer()
    index2label = {1: "label_1"}
    rel_index2type = {
        0: "top",
        1: "bottom",
        2: "center",
        3: "left",
        4: "right",
        5: "smaller",
        6: "larger",
        7: "equal",
    }
    constraint = module.TransformerSortByDictRelationConstraint(
        tokenizer,
        discrete_degree=9,
        label_set=["label_1"],
        index2label=index2label,
        rel_index2type=rel_index2type,
    )

    label_ids = [torch.tensor([1, 1])]
    relations = [
        [0, 1, 1, 1, 0],
        [0, 1, 1, 1, 1],
        [0, 1, 1, 1, 2],
        [0, 1, 1, 1, 7],
        [1, 1, 1, 2, 3],
        [1, 1, 1, 2, 4],
        [1, 1, 1, 2, 2],
        [1, 1, 1, 2, 0],
        [1, 1, 1, 2, 1],
        [1, 1, 1, 2, 5],
        [1, 1, 1, 2, 6],
        [1, 1, 1, 2, 7],
    ]
    constraint.prepare(label_ids, [relations])

    label_id = tokenizer.convert_tokens_to_ids(["label_1"])[0]
    num_id = tokenizer.convert_tokens_to_ids(["2"])[0]

    seq = [
        label_id,
        num_id,
        num_id,
        num_id,
        num_id,
        label_id,
        num_id,
        num_id,
        num_id,
        num_id,
    ]

    prev_token = torch.tensor([], dtype=torch.long)
    for seq_id, token_id in enumerate(seq):
        out, _ = constraint(0, seq_id, prev_token)
        assert len(out) > 0
        prev_token = torch.tensor([token_id])
