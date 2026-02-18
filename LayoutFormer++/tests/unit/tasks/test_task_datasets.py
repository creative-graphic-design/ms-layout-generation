from types import SimpleNamespace

import torch

from layoutformer_pp.tasks.completion import T5CompletionDataset
from layoutformer_pp.tasks.gen_r import T5GenRDataset
from layoutformer_pp.tasks.gen_t import T5GenTDataset
from layoutformer_pp.tasks.refinement import T5RefinementDataset
from layoutformer_pp.tasks.ugen import T5UGenDataset
from layoutformer_pp.tasks.task_utils import create_tokenizer
from layoutformer_pp.tasks.task_config import create_seq_processor


def _make_args():
    return SimpleNamespace(
        discrete_x_grid=16,
        discrete_y_grid=16,
        gaussian_noise_mean=0.0,
        gaussian_noise_std=0.0,
        train_bernoulli_beta=0.0,
        gen_r_discrete_before_induce_relations=False,
    )


def test_refinement_dataset_process(rico_sample):
    args = _make_args()
    tokenizer = create_tokenizer(["refinement"], "rico", discrete_grid=16)
    seq_processor = create_seq_processor("refinement")(
        dataset="rico", tokenizer=tokenizer
    )
    dataset = T5RefinementDataset(args, tokenizer, seq_processor)

    result = dataset.process(rico_sample)

    assert result["in_str"]
    assert result["out_str"]
    assert "gold_labels" in result
    assert "gold_bboxes" in result


def test_completion_dataset_process(rico_sample):
    args = _make_args()
    tokenizer = create_tokenizer(["completion"], "rico", discrete_grid=16)
    seq_processor = create_seq_processor("completion")(
        dataset="rico", tokenizer=tokenizer
    )
    dataset = T5CompletionDataset(args, tokenizer, seq_processor)

    result = dataset.process(rico_sample)

    assert len(result["input_labels"]) == 1
    assert result["in_str"]
    assert result["out_str"]


def test_ugen_dataset_process(rico_sample):
    args = _make_args()
    tokenizer = create_tokenizer(["ugen"], "rico", discrete_grid=16)
    seq_processor = create_seq_processor("ugen")(dataset="rico", tokenizer=tokenizer)
    dataset = T5UGenDataset(args, tokenizer, seq_processor)

    result = dataset.process(rico_sample)

    assert len(result["input_labels"]) == 0
    assert result["out_str"]


def test_gen_t_dataset_process(rico_sample):
    args = _make_args()
    tokenizer = create_tokenizer(["gen_t"], "rico", discrete_grid=16)
    seq_processor = create_seq_processor("gen_t")(dataset="rico", tokenizer=tokenizer)
    dataset = T5GenTDataset(args, tokenizer, seq_processor)

    result = dataset.process(rico_sample)

    assert result["in_str"]
    assert result["out_str"]
    assert torch.is_tensor(result["gold_bboxes"])


def test_gen_r_dataset_process(rico_sample):
    args = _make_args()
    tokenizer = create_tokenizer(["gen_r"], "rico", discrete_grid=16)
    seq_processor = create_seq_processor("gen_r")(dataset="rico", tokenizer=tokenizer)
    dataset = T5GenRDataset(args, tokenizer, seq_processor)

    result = dataset.process(rico_sample)

    assert result["in_str"]
    assert result["out_str"]
    relations = result["relations"]
    assert relations.numel() == 0 or relations.shape[-1] == 5
