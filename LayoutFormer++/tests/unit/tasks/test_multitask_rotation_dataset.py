from types import SimpleNamespace

from layoutformer_pp.tasks.task_config import TASK_CONFIG
from layoutformer_pp.tasks.task_utils import create_dataset, create_tokenizer


def _make_args(data_dir):
    return SimpleNamespace(
        dataset="rico",
        tasks="refinement",
        data_dir=str(data_dir),
        max_num_elements=20,
        add_sep_token=False,
        add_task_prompt=False,
        partition_training_data=False,
        fine_grained_partition_training_data=False,
        single_task_per_batch=False,
        task_weights=None,
        partition_training_data_task_buckets=None,
        fine_grained_partition_training_data_task_size=None,
        gaussian_noise_mean=0.0,
        gaussian_noise_std=0.0,
        train_bernoulli_beta=0.0,
        discrete_x_grid=16,
        discrete_y_grid=16,
        sort_by_dict=False,
        remove_too_long_layout=False,
    )


def test_rotation_dataset_sets_default_task(data_root):
    args = _make_args(data_root)
    tokenizer = create_tokenizer(["refinement"], "rico", discrete_grid=16)

    dataset = create_dataset(
        args,
        tokenizer=tokenizer,
        task_config=TASK_CONFIG,
        split="val",
        sort_by_pos=True,
    )

    assert dataset.curr_task == "refinement"
    assert dataset.seq_processor is not None
    assert hasattr(dataset.seq_processor, "index2label")
