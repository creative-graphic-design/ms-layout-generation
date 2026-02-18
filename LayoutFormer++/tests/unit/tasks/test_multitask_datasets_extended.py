from types import SimpleNamespace

import pytest

from layoutformer_pp.tasks.multitask import (
    T5MultiTaskSamplingDataset,
    T5MultiTaskConcatDataset,
    T5MultiTaskRotationDataset,
    T5MultiTaskPartitionDataset,
    T5MultiTaskFineGrainedPartitionDataset,
)
from layoutformer_pp.tasks.task_config import TASK_CONFIG
from layoutformer_pp.tasks.task_utils import create_tokenizer


def _make_args(tasks, add_task_prompt=False, task_weights=None):
    return SimpleNamespace(
        tasks=tasks,
        dataset="rico",
        add_sep_token=False,
        add_task_prompt=add_task_prompt,
        task_weights=task_weights,
        gaussian_noise_mean=0.0,
        gaussian_noise_std=0.0,
        train_bernoulli_beta=0.0,
        discrete_x_grid=16,
        discrete_y_grid=16,
    )


def _make_tokenizer(tasks):
    return create_tokenizer(tasks, "rico", discrete_grid=16)


def test_multitask_sampling_dataset_process_adds_prompt(rico_sample):
    args = _make_args("refinement,completion", add_task_prompt=True, task_weights="1,0")
    tokenizer = _make_tokenizer(["refinement", "completion"])
    dataset = T5MultiTaskSamplingDataset(
        args, tokenizer, task_config=TASK_CONFIG, sort_by_pos=True
    )

    result = dataset.process(rico_sample)

    assert result["task_name"] == "refinement"
    assert result["in_str"].startswith(TASK_CONFIG["refinement"]["prompt"])


def test_multitask_concat_dataset_getitem(rico_sample):
    args = _make_args("refinement,completion", add_task_prompt=True)
    tokenizer = _make_tokenizer(["refinement", "completion"])

    class DummyConcat(T5MultiTaskConcatDataset):
        def __init__(self, data, *args, **kwargs):
            self.data = data
            super().__init__(*args, **kwargs)

    dataset = DummyConcat(
        [rico_sample, rico_sample],
        args,
        tokenizer,
        task_config=TASK_CONFIG,
        sort_by_pos=True,
    )

    first = dataset[0]
    second = dataset[len(dataset.data)]

    assert first["task_name"] == "refinement"
    assert second["task_name"] == "completion"
    assert first["in_str"].startswith(TASK_CONFIG["refinement"]["prompt"])


def test_multitask_rotation_dataset_switch_and_error(rico_sample):
    args = _make_args("refinement,completion", add_task_prompt=False)
    tokenizer = _make_tokenizer(["refinement", "completion"])

    class DummyRotation(T5MultiTaskRotationDataset):
        def __init__(self, data, *args, **kwargs):
            self.data = data
            super().__init__(*args, **kwargs)

        def __len__(self):
            return len(self.data)

    dataset = DummyRotation(
        [rico_sample], args, tokenizer, task_config=TASK_CONFIG, sort_by_pos=True
    )

    with pytest.raises(Exception):
        dataset.switch_task("unknown-task")

    dataset.switch_task("completion")
    result = dataset[0]
    assert result["task_name"] == "completion"


def test_multitask_partition_dataset_getitem(rico_sample):
    args = _make_args("refinement,completion", add_task_prompt=False)
    tokenizer = _make_tokenizer(["refinement", "completion"])

    class DummyPartition(T5MultiTaskPartitionDataset):
        def __init__(self, data, *args, **kwargs):
            self.data = data
            super().__init__(*args, **kwargs)

        def __len__(self):
            return len(self.data)

    dataset = DummyPartition(
        [rico_sample, rico_sample, rico_sample, rico_sample],
        args,
        tokenizer,
        task_config=TASK_CONFIG,
        sort_by_pos=True,
        task_buckets="0,1",
    )

    result = dataset[0]
    assert result["task_name"] in {"refinement", "completion"}


def test_multitask_partition_dataset_bucket_mismatch(rico_sample):
    args = _make_args("refinement,completion", add_task_prompt=False)
    tokenizer = _make_tokenizer(["refinement", "completion"])

    class DummyPartition(T5MultiTaskPartitionDataset):
        def __init__(self, data, *args, **kwargs):
            self.data = data
            super().__init__(*args, **kwargs)

        def __len__(self):
            return len(self.data)

    with pytest.raises(ValueError):
        DummyPartition(
            [rico_sample, rico_sample],
            args,
            tokenizer,
            task_config=TASK_CONFIG,
            sort_by_pos=True,
            task_buckets="0",
        )


def test_multitask_fine_grained_partition_dataset(rico_sample):
    args = _make_args("refinement,completion", add_task_prompt=False)
    tokenizer = _make_tokenizer(["refinement", "completion"])

    class DummyFine(T5MultiTaskFineGrainedPartitionDataset):
        def __init__(self, data, *args, **kwargs):
            self.data = data
            super().__init__(*args, **kwargs)

        def __len__(self):
            return len(self.data)

    dataset = DummyFine(
        [rico_sample, rico_sample, rico_sample, rico_sample],
        args,
        tokenizer,
        task_config=TASK_CONFIG,
        sort_by_pos=True,
        task_data_size="1,1",
        task_weights="0.5,0.5",
    )

    result = dataset[0]
    assert result["task_name"] in {"refinement", "completion"}


def test_multitask_fine_grained_partition_dataset_size_mismatch(rico_sample):
    args = _make_args("refinement,completion", add_task_prompt=False)
    tokenizer = _make_tokenizer(["refinement", "completion"])

    class DummyFine(T5MultiTaskFineGrainedPartitionDataset):
        def __init__(self, data, *args, **kwargs):
            self.data = data
            super().__init__(*args, **kwargs)

        def __len__(self):
            return len(self.data)

    with pytest.raises(ValueError):
        DummyFine(
            [rico_sample, rico_sample],
            args,
            tokenizer,
            task_config=TASK_CONFIG,
            sort_by_pos=True,
            task_data_size="1",
            task_weights=None,
        )
