import torch

from layoutformer_pp.model.layout_transformer.tokenizer import (
    LayoutTransformerTokenizer,
)
from layoutformer_pp.tasks.refinement import T5LayoutSequence, refinement_inference


class DummyModel:
    def __init__(self, output_ids: torch.Tensor) -> None:
        self.output_ids = output_ids

    def __call__(
        self,
        input_ids,
        padding_mask,
        max_length=0,
        generation_constraint_fn=None,
        task_ids=None,
        do_sample=False,
        top_k=10,
        temperature=0.7,
        constrained_decoding=False,
    ):
        batch = input_ids.size(0)
        output = self.output_ids.expand(batch, -1)
        return {"output": output}


def test_parse_then_place_pipeline_roundtrip():
    tokens = ["label1", "label2", "0", "1", "2", "3", "4", "5", "6", "7", "8"]
    tokenizer = LayoutTransformerTokenizer(tokens)
    index2label = {1: "label1", 2: "label2"}
    label2index = {"label1": 1, "label2": 2}
    seq_processor = T5LayoutSequence(tokenizer, index2label, label2index)

    labels = torch.tensor([1, 2])
    bboxes = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])

    seq = seq_processor.build_seq(labels, bboxes).lower().strip()
    output_ids = tokenizer(seq, add_eos=True, add_bos=False)["input_ids"]
    model = DummyModel(output_ids)

    data = {
        "gold_labels": [labels],
        "gold_bboxes": [bboxes],
        "input_labels": [labels],
        "input_bboxes": [bboxes],
        "in_str": [seq],
        "out_str": [seq],
        "name": ["sample"],
    }

    metric, out = refinement_inference(
        model,
        data,
        seq_processor,
        tokenizer,
        device="cpu",
    )

    assert metric["num_examples"] == 1
    assert metric["num_label_correct"].item() == 1
    assert metric["num_bbox_correct"].item() == 8
    assert out["pred_labels"].shape == out["gold_labels"].shape
    assert out["pred_bboxes"].shape == out["gold_bboxes"].shape
