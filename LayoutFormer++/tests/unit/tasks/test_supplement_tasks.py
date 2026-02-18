from types import SimpleNamespace

import torch
import pytest

from layoutformer_pp.data import PubLayNetDataset
from layoutformer_pp.model.layout_transformer.tokenizer import (
    LayoutTransformerTokenizer,
)
from layoutformer_pp.model.utils import TransformerSortByDictRelationConstraint
from layoutformer_pp.tasks.gen_r import RelationTypes
from layoutformer_pp.tasks.supplement import gen_tc, gen_tsc, gen_rp, gen_rs
from layoutformer_pp.utils import utils


class DummyModel:
    def __init__(self, tokenizer, output_strings):
        self.tokenizer = tokenizer
        self.output_strings = output_strings

    def __call__(self, in_ids, in_padding_mask, max_length=0, **kwargs):
        tokenized = self.tokenizer(self.output_strings, add_eos=True, add_bos=False)
        return {"output": tokenized["input_ids"]}


def _make_tokenizer(discrete_degree, include_relations=False):
    tokens = list(PubLayNetDataset.labels)
    tokens.extend(map(str, range(discrete_degree)))
    tokens.append(gen_tc.T5LayoutSequenceForGenTC.SEP_ELE_TYPE_TOKEN)

    if include_relations:
        tokens.append("label_0")
        tokens.extend([f"label_{i + 1}" for i in range(len(PubLayNetDataset.labels))])
        tokens.extend([f"index_{i}" for i in range(1, 6)])
        tokens.extend([f"relation_{i}" for i in range(len(RelationTypes.types))])
        tokens.extend(
            [
                gen_rp.T5LayoutSequenceForGenRP.REL_BEG_TOKEN,
                gen_rp.T5LayoutSequenceForGenRP.REL_SEP_TOKEN,
                gen_rp.T5LayoutSequenceForGenRP.REL_ELE_SEP_TOKEN,
            ]
        )

    return LayoutTransformerTokenizer(tokens)


@pytest.mark.parametrize(
    "seq_cls,dataset_cls,infer_fn,seq_kwargs",
    [
        (
            gen_tc.T5LayoutSequenceForGenTC,
            gen_tc.T5GenTCDataset,
            gen_tc.gen_tc_inference,
            {"gen_t_add_unk_token": True},
        ),
        (
            gen_tsc.T5LayoutSequenceForGenTSC,
            gen_tsc.T5GenTSCDataset,
            gen_tsc.gen_tsc_inference,
            {"label_size_add_unk_token": True},
        ),
    ],
)
def test_gen_tc_and_tsc_process_and_inference(
    publaynet_sample, seq_cls, dataset_cls, infer_fn, seq_kwargs
):
    args = SimpleNamespace(discrete_x_grid=10, discrete_y_grid=10)
    index2label = PubLayNetDataset.index2label(PubLayNetDataset.labels)
    label2index = PubLayNetDataset.label2index(PubLayNetDataset.labels)

    tokenizer = _make_tokenizer(discrete_degree=10, include_relations=False)
    seq_processor = seq_cls(
        tokenizer, index2label, label2index, add_sep_token=False, **seq_kwargs
    )
    dataset = dataset_cls(args, tokenizer, seq_processor)

    processed = dataset.process(publaynet_sample)
    assert seq_processor.SEP_ELE_TYPE_TOKEN in processed["in_str"]

    data = utils.collate_fn([processed])
    model = DummyModel(tokenizer, data["out_str"])

    metric, out = infer_fn(
        model, data, seq_processor, tokenizer, device=torch.device("cpu")
    )

    assert "pred_labels" in out
    assert "pred_bboxes" in out
    assert metric["num_examples"] == 1


@pytest.mark.parametrize("discrete_first", [False, True])
@pytest.mark.parametrize(
    "seq_cls,dataset_cls,infer_fn",
    [
        (
            gen_rp.T5LayoutSequenceForGenRP,
            gen_rp.T5GenRPDataset,
            gen_rp.gen_rp_inference,
        ),
        (
            gen_rs.T5LayoutSequenceForGenRS,
            gen_rs.T5GenRSDataset,
            gen_rs.gen_rs_inference,
        ),
    ],
)
def test_gen_rp_rs_process_and_inference(
    publaynet_sample, seq_cls, dataset_cls, infer_fn, discrete_first
):
    args = SimpleNamespace(
        discrete_x_grid=10,
        discrete_y_grid=10,
        gen_r_discrete_before_induce_relations=discrete_first,
    )
    index2label = PubLayNetDataset.index2label(PubLayNetDataset.labels)
    label2index = PubLayNetDataset.label2index(PubLayNetDataset.labels)

    tokenizer = _make_tokenizer(discrete_degree=10, include_relations=True)
    seq_processor = seq_cls(
        tokenizer,
        index2label,
        label2index,
        add_sep_token=False,
        gen_r_add_unk_token=True,
        gen_r_compact=False,
        discrete_x_grid=10,
        discrete_y_grid=10,
    )
    dataset = dataset_cls(args, tokenizer, seq_processor)

    processed = dataset.process(publaynet_sample)
    data = utils.collate_fn([processed])

    model = DummyModel(tokenizer, data["out_str"])
    constraint = TransformerSortByDictRelationConstraint(
        tokenizer,
        discrete_degree=10,
        label_set=list(index2label.values()),
        index2label=index2label,
        rel_index2type=RelationTypes.index2type(),
    )

    metric, out = infer_fn(
        model,
        data,
        seq_processor,
        tokenizer,
        device=torch.device("cpu"),
        constraint_fn=constraint,
    )

    assert "relations" in out
    assert metric["num_examples"] >= 1


def test_gen_rp_format_and_violation():
    result = {"pred": (torch.tensor([[0.1, 0.1, 0.2, 0.2]]), torch.tensor([1]))}
    layout = gen_rp.format_gen_r_pred_layout(result)

    assert layout["labels_with_canvas"][0].item() == 0

    relations = torch.tensor([[0, 1, 1, 1, RelationTypes.type2index()["left"]]])
    violations = gen_rp.compute_rel_violation(
        layout, relations, RelationTypes.type2index()
    )
    assert isinstance(violations, int)

    bad_relations = torch.tensor([[99, 1, 1, 1, 0]])
    assert (
        gen_rp.compute_rel_violation(layout, bad_relations, RelationTypes.type2index())
        >= 1
    )
