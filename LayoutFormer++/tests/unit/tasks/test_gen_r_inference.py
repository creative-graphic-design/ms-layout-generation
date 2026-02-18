from types import SimpleNamespace

import pytest
import torch

from layoutformer_pp.data import transforms
from layoutformer_pp.model.layout_transformer.constrained_decoding import (
    TransformerSortByDictRelationConstraint,
)
from layoutformer_pp.tasks.gen_r import (
    AddCanvasElement,
    AddRelation,
    RelationTypes,
    T5GenRDataset,
    detect_loc_relation,
    detect_size_relation,
    gen_r_inference,
    get_label_with_index,
)
from layoutformer_pp.tasks.task_config import create_seq_processor
from layoutformer_pp.tasks.task_utils import create_tokenizer
from layoutformer_pp.utils import utils


class DummyModel:
    def __init__(self, tokenizer, output_strings):
        self.tokenizer = tokenizer
        self.output_strings = output_strings

    def __call__(self, in_ids, in_padding_mask, max_length=0, **kwargs):
        tokenized = self.tokenizer(self.output_strings, add_eos=True, add_bos=False)
        return {"output": tokenized["input_ids"]}


def _make_args():
    return SimpleNamespace(
        discrete_x_grid=16,
        discrete_y_grid=16,
        gen_r_discrete_before_induce_relations=False,
    )


def test_add_canvas_element_and_relation():
    discrete_fn = transforms.DiscretizeBoundingBox(num_x_grid=10, num_y_grid=10)
    data = {
        "discrete_gold_bboxes": torch.tensor([[1, 1, 2, 2]]),
        "bboxes": torch.tensor([[0.1, 0.1, 0.2, 0.2]]),
        "labels": torch.tensor([1]),
    }
    add_canvas = AddCanvasElement(use_discrete=True, discrete_fn=discrete_fn)
    out = add_canvas(data)
    assert out["bboxes_with_canvas"].shape[0] == 2
    assert out["labels_with_canvas"][0].item() == 0

    add_rel = AddRelation(seed=1, ratio=1.0)
    rel_out = add_rel(out)
    assert "relations" in rel_out
    assert rel_out["relations"].ndim == 2


def test_detect_size_relation_branches():
    large = torch.tensor([0.0, 0.0, 2.0, 2.0])
    small = torch.tensor([0.0, 0.0, 1.0, 1.0])
    assert detect_size_relation(large, small) == "smaller"
    assert detect_size_relation(small, small) == "equal"
    assert detect_size_relation(small, large) == "larger"


@pytest.mark.parametrize(
    "bbox,expected",
    [
        (torch.tensor([0.0, 0.0, 1.0, 0.2]), "top"),
        (torch.tensor([0.0, 0.4, 1.0, 0.2]), "center"),
        (torch.tensor([0.0, 0.8, 1.0, 0.2]), "bottom"),
    ],
)
def test_detect_loc_relation_canvas(bbox, expected):
    assert (
        detect_loc_relation(torch.tensor([0.0, 0.0, 1.0, 1.0]), bbox, canvas=True)
        == expected
    )


@pytest.mark.parametrize(
    "b1,b2,expected",
    [
        (
            torch.tensor([0.0, 0.0, 1.0, 1.0]),
            torch.tensor([0.0, -2.0, 1.0, 1.0]),
            "top",
        ),
        (
            torch.tensor([0.0, 0.0, 1.0, 1.0]),
            torch.tensor([0.0, 2.0, 1.0, 1.0]),
            "bottom",
        ),
        (
            torch.tensor([1.0, 0.0, 1.0, 1.0]),
            torch.tensor([0.0, 0.0, 1.0, 1.0]),
            "left",
        ),
        (
            torch.tensor([0.0, 0.0, 1.0, 1.0]),
            torch.tensor([2.0, 0.0, 1.0, 1.0]),
            "right",
        ),
        (
            torch.tensor([0.0, 0.0, 1.0, 1.0]),
            torch.tensor([0.2, 0.2, 0.5, 0.5]),
            "center",
        ),
    ],
)
def test_detect_loc_relation_non_canvas(b1, b2, expected):
    assert detect_loc_relation(b1, b2, canvas=False) == expected


def test_get_label_with_index():
    labels = torch.tensor([1, 1, 2, 1])
    assert get_label_with_index(labels) == [1, 2, 1, 3]


def test_gen_r_inference_with_constraint_and_violation(rico_sample):
    args = _make_args()
    tokenizer = create_tokenizer(["gen_r"], "rico", discrete_grid=16)
    seq_processor = create_seq_processor("gen_r")(
        dataset="rico",
        tokenizer=tokenizer,
        gen_r_add_unk_token=False,
        gen_r_compact=False,
        discrete_x_grid=16,
        discrete_y_grid=16,
    )
    dataset = T5GenRDataset(args, tokenizer, seq_processor)

    processed = dataset.process(rico_sample)
    data = utils.collate_fn([processed, processed, processed])
    data["task_id"] = [5, 5, 5]

    custom_rel = torch.tensor([[1, 1, 1, 1, RelationTypes.type2index()["left"]]]).long()
    data["relations"] = [custom_rel, custom_rel, custom_rel]
    data["in_str"] = [
        seq_processor.build_input_seq(processed["gold_labels"], custom_rel)
        .lower()
        .strip()
    ] * 3

    short_str = (
        seq_processor.build_seq(
            processed["gold_labels"][:1], processed["gold_bboxes"][:1]
        )
        .lower()
        .strip()
    )
    long_str = f"{processed['out_str']} {processed['out_str']}"

    output_strings = ["bad output", short_str, long_str]
    model = DummyModel(tokenizer, output_strings)

    index2label = seq_processor.index2label
    constraint_fn = TransformerSortByDictRelationConstraint(
        tokenizer,
        discrete_degree=16,
        label_set=list(index2label.values()),
        index2label=index2label,
        rel_index2type=RelationTypes.index2type(),
        add_sep_token=False,
    )

    metric, out = gen_r_inference(
        model,
        data,
        seq_processor,
        tokenizer,
        device=torch.device("cpu"),
        constraint_fn=constraint_fn,
    )

    assert metric["num_examples"] == 3
    assert out["pred_bboxes"].shape[-1] == 4
    assert len(out["violate_num"]) == 3


def test_gen_r_inference_non_relation_constraint(rico_sample):
    args = _make_args()
    tokenizer = create_tokenizer(["gen_r"], "rico", discrete_grid=16)
    seq_processor = create_seq_processor("gen_r")(
        dataset="rico",
        tokenizer=tokenizer,
        gen_r_add_unk_token=False,
        gen_r_compact=False,
        discrete_x_grid=16,
        discrete_y_grid=16,
    )
    dataset = T5GenRDataset(args, tokenizer, seq_processor)
    processed = dataset.process(rico_sample)
    data = utils.collate_fn([processed])
    data["task_id"] = [5]

    class DummyConstraint:
        def __init__(self):
            self.prepared = False

        def prepare(self, labels):
            self.prepared = True

    constraint = DummyConstraint()
    model = DummyModel(tokenizer, data["out_str"])

    gen_r_inference(
        model,
        data,
        seq_processor,
        tokenizer,
        device=torch.device("cpu"),
        constraint_fn=constraint,
    )

    assert constraint.prepared is True
