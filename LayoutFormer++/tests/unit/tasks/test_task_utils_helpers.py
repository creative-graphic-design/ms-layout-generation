from types import SimpleNamespace

import torch
import pytest

from layoutformer_pp.tasks import task_utils
from layoutformer_pp.model import LayoutTransformerTokenizer
from layoutformer_pp.tasks.task_config import create_seq_processor


class DummyModel:
    def __init__(self):
        self.last_loss_weights = None

    def __call__(self, in_ids, padding_mask, out_ids, loss_weights=None, task_ids=None):
        self.last_loss_weights = loss_weights
        return {"loss": torch.ones(in_ids.size(0))}


class DummySeq1:
    def parse_seq(self, output_str):
        return [1], [[0, 0, 1, 1]]


class DummySeq2:
    def parse_seq(self, input_str, output_str):
        return [1], [[0, 0, 1, 1]]


def _make_args():
    return SimpleNamespace(
        add_task_prompt=False,
        add_task_prompt_token_in_model=False,
        d_model=8,
        num_layers=1,
        nhead=2,
        dropout=0.0,
        share_embedding=False,
        add_task_embedding=False,
        num_task_prompt_token=2,
        num_pos_embed=16,
        max_num_elements=5,
        dataset="publaynet",
        partition_training_data=False,
        fine_grained_partition_training_data=False,
        single_task_per_batch=False,
        remove_too_long_layout=False,
        partition_training_data_task_buckets=1,
        fine_grained_partition_training_data_task_size=1,
        task_weights=None,
    )


def test_create_tokenizer_includes_relations():
    tokenizer = task_utils.create_tokenizer(
        tasks=["gen_r"],
        dataset="publaynet",
        discrete_grid=5,
        add_sep_token=True,
    )
    token_ids = tokenizer.convert_tokens_to_ids(["label_0", "relation_0", "index_1"])
    assert all(tid != tokenizer.unk_token_id for tid in token_ids)


def test_build_model_mutually_exclusive_prompt_tokens():
    args = _make_args()
    args.add_task_prompt = True
    args.add_task_prompt_token_in_model = True
    tokenizer = task_utils.create_tokenizer(
        tasks=["refinement"],
        dataset="publaynet",
        discrete_grid=5,
    )
    with pytest.raises(TypeError):
        task_utils.build_model(args, tokenizer)


def test_train_fn_applies_task_loss_weights():
    tokenizer = task_utils.create_tokenizer(
        tasks=["refinement"],
        dataset="publaynet",
        discrete_grid=5,
    )
    model = DummyModel()
    train_fn = task_utils.TrainFn(task_loss_weights={"refinement": 2.0})

    data = {
        "in_str": ["label_1 0 0 1 1"],
        "out_str": ["label_1 0 0 1 1"],
        "task_name": ["refinement"],
    }

    loss = train_fn(model, data, tokenizer, device=torch.device("cpu"))

    assert loss.item() == pytest.approx(1.0)
    assert model.last_loss_weights is not None


def test_evaluate_fn_parse_seq_handles_variants():
    eval_fn = task_utils.EvaluateFn(max_num_elements=5)

    labels, bboxes = eval_fn._parse_seq(DummySeq1(), "out")
    assert labels == [1]
    assert bboxes == [[0, 0, 1, 1]]

    labels, bboxes = eval_fn._parse_seq(DummySeq2(), "out")
    assert labels == [1]
    assert bboxes == [[0, 0, 1, 1]]


def test_create_tokenizer_adds_task_prompt_tokens():
    tokenizer = task_utils.create_tokenizer(
        tasks=["refinement", "gen_r"],
        dataset="rico",
        discrete_grid=5,
        add_task_prompt=True,
    )
    token_ids = tokenizer.convert_tokens_to_ids(["task-refinement", "task-gen_r"])
    assert all(tid != tokenizer.unk_token_id for tid in token_ids)


def test_create_tokenizer_invalid_dataset_raises():
    with pytest.raises(NotImplementedError):
        task_utils.create_tokenizer(
            tasks=["refinement"],
            dataset="unknown",
            discrete_grid=5,
        )


def test_create_dataset_invalid_dataset_raises():
    args = _make_args()
    args.dataset = "unknown"
    tokenizer = LayoutTransformerTokenizer(tokens=["label_1", "0"])
    with pytest.raises(NotImplementedError):
        task_utils.create_dataset(args, tokenizer, task_config={}, split="train")


def test_create_seq_processor_supplement_tasks():
    tokenizer = LayoutTransformerTokenizer(tokens=["label_1", "0"])
    for task in ["gen_tc", "gen_tsc", "gen_rs", "gen_rp"]:
        seq_proc_fn = create_seq_processor(task)
        seq_proc = seq_proc_fn("rico", tokenizer)
        assert hasattr(seq_proc, "build_seq")
