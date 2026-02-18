from types import SimpleNamespace

import torch

from layoutformer_pp.tasks.task_utils import (
    TrainFn,
    EvaluateFn,
    build_model,
    create_dataset,
    create_tokenizer,
)
from layoutformer_pp.tasks.task_config import TASK_CONFIG
from layoutformer_pp.utils import utils


def _make_args(data_root):
    return SimpleNamespace(
        dataset="rico",
        tasks="refinement",
        data_dir=str(data_root),
        max_num_elements=20,
        remove_too_long_layout=False,
        partition_training_data=False,
        partition_training_data_task_buckets=None,
        fine_grained_partition_training_data=False,
        fine_grained_partition_training_data_task_size=None,
        task_weights=None,
        task_loss_weights=None,
        single_task_per_batch=False,
        add_sep_token=False,
        add_task_prompt=False,
        sort_by_dict=False,
        discrete_x_grid=16,
        discrete_y_grid=16,
        gaussian_noise_mean=0.0,
        gaussian_noise_std=0.0,
        train_bernoulli_beta=0.0,
        d_model=32,
        num_layers=2,
        nhead=4,
        dropout=0.1,
        share_embedding=False,
        num_pos_embed=128,
        add_task_embedding=False,
        add_task_prompt_token_in_model=False,
        num_task_prompt_token=1,
        eval_seed=123,
        decode_max_length=64,
        topk=5,
        temperature=0.7,
        enable_task_measure=False,
    )


def test_train_and_eval_pipeline(data_root):
    args = _make_args(data_root)
    tokenizer = create_tokenizer(["refinement"], args.dataset, args.discrete_x_grid)
    model = build_model(args, tokenizer)

    dataset = create_dataset(
        args,
        tokenizer=tokenizer,
        task_config=TASK_CONFIG,
        split="train",
        sort_by_pos=True,
    )

    sample = dataset[0]
    batch = utils.collate_fn([sample])

    train_fn = TrainFn()
    loss = train_fn(model, batch, tokenizer, device=torch.device("cpu"))
    assert loss.dim() == 0

    eval_fn = EvaluateFn(
        args.max_num_elements,
        enable_task_measure=False,
        decode_max_length=args.decode_max_length,
        topk=args.topk,
        temperature=args.temperature,
    )
    seq_processor = dataset.seq_processor["refinement"]
    eval_loss, prediction = eval_fn(
        model,
        batch,
        seq_processor=seq_processor,
        tokenizer=tokenizer,
        device=torch.device("cpu"),
    )
    assert eval_loss.dim() == 0
    assert prediction is None
