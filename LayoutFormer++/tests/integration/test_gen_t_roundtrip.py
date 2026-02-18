import torch

from layoutformer_pp.model.layout_transformer.tokenizer import (
    LayoutTransformerTokenizer,
)
from layoutformer_pp.tasks.gen_t import T5LayoutSequenceForGenT
from layoutformer_pp.evaluation import metrics


def test_gen_t_sequence_roundtrip_and_metrics():
    tokens = ["label1", "label2", "0", "1", "2", "3", "4", "5", "6", "7", "8"]
    tokenizer = LayoutTransformerTokenizer(tokens)
    index2label = {1: "label1", 2: "label2"}
    label2index = {"label1": 1, "label2": 2}

    seq_processor = T5LayoutSequenceForGenT(
        T5LayoutSequenceForGenT.GEN_T,
        tokenizer,
        index2label,
        label2index,
    )

    labels = torch.tensor([1, 2])
    bboxes = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])

    out_str = seq_processor.build_seq(labels, bboxes).lower().strip()
    parsed_labels, parsed_bbox = seq_processor.parse_seq(out_str)

    assert parsed_labels == [1, 2]
    assert parsed_bbox == [[1, 2, 3, 4], [4, 3, 2, 1]]

    pred_labels = torch.tensor([parsed_labels])
    gold_labels = labels.unsqueeze(0)
    mask = torch.tensor([[True, True]])

    num_label_correct, num_examples = metrics.calculate_label_accuracy(
        gold_labels, pred_labels, mask, element_wise=False
    )
    assert num_label_correct.item() == 1
    assert num_examples == 1

    pred_bbox = torch.tensor([parsed_bbox])
    gold_bbox = bboxes.unsqueeze(0)
    num_bbox_correct, num_bbox_total = metrics.calculate_bbox_accuracy(
        gold_bbox, pred_bbox, mask, element_wise=False
    )
    assert num_bbox_correct.item() == 8
    assert num_bbox_total.item() == 8
