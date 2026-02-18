from types import SimpleNamespace

import torch

from layoutformer_pp.tasks.completion import T5CompletionDataset, completion_inference
from layoutformer_pp.tasks.gen_t import T5GenTDataset, gen_t_inference
from layoutformer_pp.tasks.ugen import T5UGenDataset, ugen_inference
from layoutformer_pp.tasks.task_config import create_seq_processor
from layoutformer_pp.tasks.task_utils import create_tokenizer
from layoutformer_pp.utils import utils
from layoutformer_pp.model.layout_transformer.constrained_decoding import (
    TransformerSortByDictLabelSizeConstraint,
)


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
        gaussian_noise_mean=0.0,
        gaussian_noise_std=0.0,
        train_bernoulli_beta=0.0,
    )


def test_completion_inference_handles_invalid_outputs(rico_sample):
    args = _make_args()
    tokenizer = create_tokenizer(["completion"], "rico", discrete_grid=16)
    seq_processor = create_seq_processor("completion")(
        dataset="rico", tokenizer=tokenizer
    )
    dataset = T5CompletionDataset(args, tokenizer, seq_processor)

    processed = dataset.process(rico_sample)
    data = utils.collate_fn([processed, processed])
    data["task_id"] = [1, 1]

    output_strings = ["bad output", data["out_str"][1]]
    model = DummyModel(tokenizer, output_strings)

    metric, out = completion_inference(
        model, data, seq_processor, tokenizer, device=torch.device("cpu")
    )

    assert metric["num_examples"] == 2
    assert out["gold_labels"].shape[0] == 2
    assert out["pred_labels"].ndim == 2


def test_ugen_inference_skips_invalid_outputs(rico_sample):
    args = _make_args()
    tokenizer = create_tokenizer(["ugen"], "rico", discrete_grid=16)
    seq_processor = create_seq_processor("ugen")(dataset="rico", tokenizer=tokenizer)
    dataset = T5UGenDataset(args, tokenizer, seq_processor)

    processed = dataset.process(rico_sample)
    data = utils.collate_fn([processed, processed])
    data["task_id"] = [2, 2]

    output_strings = ["bad output", data["out_str"][1]]
    model = DummyModel(tokenizer, output_strings)

    metric, out = ugen_inference(
        model, data, seq_processor, tokenizer, device=torch.device("cpu")
    )

    assert metric["num_examples"] == 2
    assert out["gold_labels"].shape[0] == 2
    assert out["pred_labels"].ndim == 2


def test_gen_t_inference_padding_and_truncation(rico_sample):
    args = _make_args()
    tokenizer = create_tokenizer(["gen_t"], "rico", discrete_grid=16)
    seq_processor = create_seq_processor("gen_t")(dataset="rico", tokenizer=tokenizer)
    dataset = T5GenTDataset(args, tokenizer, seq_processor)

    processed = dataset.process(rico_sample)
    data = utils.collate_fn([processed, processed, processed])
    data["task_id"] = [3, 3, 3]

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
    constraint_fn = TransformerSortByDictLabelSizeConstraint(
        tokenizer,
        discrete_degree=16,
        label_set=list(index2label.values()),
        index2label=index2label,
        add_sep_token=False,
    )

    metric, out = gen_t_inference(
        model,
        data,
        seq_processor,
        tokenizer,
        device=torch.device("cpu"),
        constraint_fn=constraint_fn,
    )

    assert metric["num_examples"] == 3
    assert out["pred_labels"].shape[0] == 3
    assert out["pred_bboxes"].shape[-1] == 4


def test_gen_t_inference_non_size_constraint(rico_sample):
    args = _make_args()
    tokenizer = create_tokenizer(["gen_t"], "rico", discrete_grid=16)
    seq_processor = create_seq_processor("gen_t")(dataset="rico", tokenizer=tokenizer)
    dataset = T5GenTDataset(args, tokenizer, seq_processor)

    processed = dataset.process(rico_sample)
    data = utils.collate_fn([processed])
    data["task_id"] = [3]

    model = DummyModel(tokenizer, data["out_str"])

    class DummyConstraint:
        def __init__(self):
            self.prepared = False

        def prepare(self, labels):
            self.prepared = True

    constraint = DummyConstraint()

    gen_t_inference(
        model,
        data,
        seq_processor,
        tokenizer,
        device=torch.device("cpu"),
        constraint_fn=constraint,
    )

    assert constraint.prepared is True
