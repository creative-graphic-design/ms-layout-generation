import torch

from layoutformer_pp.model.layout_transformer.tokenizer import (
    LayoutTransformerTokenizer,
)
from layoutformer_pp.tasks.refinement import T5LayoutSequence
from layoutformer_pp.tasks.completion import T5CompletionLayoutSequence
from layoutformer_pp.tasks.gen_t import T5LayoutSequenceForGenT
from layoutformer_pp.utils import utils


def _make_tokenizer():
    tokens = [
        "label1",
        "label2",
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "|",
    ]
    return LayoutTransformerTokenizer(tokens)


def _make_seq_processor(add_sep_token: bool = False):
    tokenizer = _make_tokenizer()
    index2label = {1: "label1", 2: "label2"}
    label2index = {"label1": 1, "label2": 2}
    return tokenizer, T5LayoutSequence(
        tokenizer, index2label, label2index, add_sep_token=add_sep_token
    )


def test_parse_predicted_layout_valid_and_invalid():
    labels, bbox = utils.parse_predicted_layout("text button 1 2 3 4 image 5 6 7 8")
    assert labels == ["text button", "image"]
    assert bbox == [[1, 2, 3, 4], [5, 6, 7, 8]]

    assert utils.parse_predicted_layout("text 1 2 3") == (None, None)


def test_t5layoutsequence_roundtrip_without_sep():
    tokenizer, seq_processor = _make_seq_processor(add_sep_token=False)
    labels = torch.tensor([1, 2])
    bboxes = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])

    seq = seq_processor.build_seq(labels, bboxes).lower().strip()
    parsed_labels, parsed_bbox = seq_processor.parse_seq(seq)

    assert parsed_labels == [1, 2]
    assert parsed_bbox == [[1, 2, 3, 4], [5, 6, 7, 8]]
    assert (
        tokenizer.decode(
            tokenizer(seq)["input_ids"][0].tolist(), skip_special_tokens=True
        ).strip()
        == seq
    )


def test_t5layoutsequence_roundtrip_with_sep():
    _, seq_processor = _make_seq_processor(add_sep_token=True)
    labels = torch.tensor([1, 2])
    bboxes = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])

    seq = seq_processor.build_seq(labels, bboxes).lower().strip()
    assert "|" in seq

    parsed_labels, parsed_bbox = seq_processor.parse_seq(seq)
    assert parsed_labels == [1, 2]
    assert parsed_bbox == [[1, 2, 3, 4], [5, 6, 7, 8]]


def test_completion_sequence_parse_signature():
    tokenizer = _make_tokenizer()
    index2label = {1: "label1"}
    label2index = {"label1": 1}
    seq_processor = T5CompletionLayoutSequence(tokenizer, index2label, label2index)

    seq = "label1 1 2 3 4"
    labels, bbox = seq_processor.parse_seq(None, seq)

    assert labels == [1]
    assert bbox == [[1, 2, 3, 4]]


def test_gen_t_build_input_seq_with_unk_tokens():
    tokenizer = _make_tokenizer()
    index2label = {1: "label1"}
    label2index = {"label1": 1}

    seq_processor = T5LayoutSequenceForGenT(
        T5LayoutSequenceForGenT.GEN_T,
        tokenizer,
        index2label,
        label2index,
        gen_t_add_unk_token=True,
    )

    labels = torch.tensor([1])
    bboxes = torch.tensor([[1, 2, 3, 4]])
    seq = seq_processor.build_input_seq(labels, bboxes)

    expected_tokens = [
        "label1",
        tokenizer.unk_token,
        tokenizer.unk_token,
        tokenizer.unk_token,
        tokenizer.unk_token,
    ]
    assert seq.split() == expected_tokens


def test_gen_ts_build_input_seq_with_size_and_unk_tokens():
    tokenizer = _make_tokenizer()
    index2label = {1: "label1"}
    label2index = {"label1": 1}

    seq_processor = T5LayoutSequenceForGenT(
        T5LayoutSequenceForGenT.GEN_TS,
        tokenizer,
        index2label,
        label2index,
        gen_ts_add_unk_token=True,
    )

    labels = torch.tensor([1])
    bboxes = torch.tensor([[1, 2, 3, 4]])
    seq = seq_processor.build_input_seq(labels, bboxes)

    expected_tokens = [
        "label1",
        tokenizer.unk_token,
        tokenizer.unk_token,
        "3",
        "4",
    ]
    assert seq.split() == expected_tokens
