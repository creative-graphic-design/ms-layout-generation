import torch

from layoutformer_pp.data.transforms import DiscretizeBoundingBox
from layoutformer_pp.tasks.gen_r import (
    RelationTypes,
    T5LayoutSequenceForGenR,
    detect_loc_relation,
    detect_size_relation,
)
from layoutformer_pp.tasks.gen_t import T5LayoutSequenceForGenT
from layoutformer_pp.tasks.refinement import T5LayoutSequence
from layoutformer_pp.tasks.task_utils import create_tokenizer
from layoutformer_pp.tasks.task_config import create_seq_processor
from layoutformer_pp.data import RicoDataset


def _make_tokenizer():
    return create_tokenizer(["refinement"], "rico", discrete_grid=16)


def test_t5layoutsequence_roundtrip(rico_sample):
    tokenizer = _make_tokenizer()
    index2label = RicoDataset.index2label(RicoDataset.labels)
    label2index = RicoDataset.label2index(RicoDataset.labels)
    seq = T5LayoutSequence(tokenizer, index2label, label2index)

    discretizer = DiscretizeBoundingBox(16, 16)
    labels = rico_sample["labels"][:2]
    bboxes = discretizer.discretize(rico_sample["bboxes"][:2])

    text = seq.build_seq(labels, bboxes)
    parsed_labels, parsed_bboxes = seq.parse_seq(text.lower())

    assert parsed_labels == labels.tolist()
    assert parsed_bboxes == bboxes.tolist()


def test_t5layoutsequence_with_sep_token(rico_sample):
    tokenizer = _make_tokenizer()
    index2label = RicoDataset.index2label(RicoDataset.labels)
    label2index = RicoDataset.label2index(RicoDataset.labels)
    seq = T5LayoutSequence(tokenizer, index2label, label2index, add_sep_token=True)

    discretizer = DiscretizeBoundingBox(16, 16)
    labels = rico_sample["labels"][:2]
    bboxes = discretizer.discretize(rico_sample["bboxes"][:2])

    text = seq.build_seq(labels, bboxes)
    assert "|" in text

    parsed_labels, parsed_bboxes = seq.parse_seq(text.lower())
    assert parsed_labels == labels.tolist()
    assert parsed_bboxes == bboxes.tolist()


def test_gen_t_and_gen_ts_input_sequences(rico_sample):
    tokenizer = create_tokenizer(["gen_t"], "rico", discrete_grid=16)
    index2label = RicoDataset.index2label(RicoDataset.labels)
    label2index = RicoDataset.label2index(RicoDataset.labels)
    discretizer = DiscretizeBoundingBox(16, 16)
    labels = rico_sample["labels"][:2]
    bboxes = discretizer.discretize(rico_sample["bboxes"][:2])

    gen_t_seq = T5LayoutSequenceForGenT(
        T5LayoutSequenceForGenT.GEN_T,
        tokenizer,
        index2label,
        label2index,
        add_sep_token=True,
    )
    gen_ts_seq = T5LayoutSequenceForGenT(
        T5LayoutSequenceForGenT.GEN_TS,
        tokenizer,
        index2label,
        label2index,
        add_sep_token=True,
    )

    gen_t_in = gen_t_seq.build_input_seq(labels, bboxes)
    gen_ts_in = gen_ts_seq.build_input_seq(labels, bboxes)

    assert "|" in gen_t_in
    assert "|" in gen_ts_in
    assert str(bboxes[0][2].item()) in gen_ts_in


def test_gen_r_detect_relations():
    b1 = torch.tensor([0.0, 0.0, 0.2, 0.2])
    b2 = torch.tensor([0.0, 0.5, 0.2, 0.2])

    assert detect_size_relation(b1, b2) in RelationTypes.types
    assert detect_loc_relation(b1, b2, canvas=False) == "bottom"
    assert detect_loc_relation(b1, b2, canvas=True) in {"top", "center", "bottom"}


def test_gen_r_input_sequence_includes_relation_tokens(rico_sample):
    tokenizer = create_tokenizer(["gen_r"], "rico", discrete_grid=16)
    seq_processor = create_seq_processor("gen_r")(dataset="rico", tokenizer=tokenizer)
    labels = rico_sample["labels"][:2]
    relations = torch.tensor([[0, 1, labels[0].item(), 1, 0]])

    in_seq = seq_processor.build_input_seq(labels, relations)
    assert T5LayoutSequenceForGenR.REL_BEG_TOKEN in in_seq
    assert "relation_0" in in_seq
