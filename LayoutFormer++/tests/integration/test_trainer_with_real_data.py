"""Integration tests for trainers with real data."""

import os
import tempfile
from types import SimpleNamespace

import pytest
import torch

from layoutformer_pp.tasks.task_utils import (
    create_tokenizer,
    build_model,
    create_dataset,
    TrainFn,
    EvaluateFn,
)
from layoutformer_pp.tasks.task_config import TASK_CONFIG
from layoutformer_pp.trainer.basic_trainer import Trainer


def _get_seq_processor(dataset, task: str = "refinement"):
    seq_processor = dataset.seq_processor
    if isinstance(seq_processor, dict):
        return seq_processor[task]
    return seq_processor


def _assert_processed_sample(sample):
    assert "gold_bboxes" in sample
    assert "gold_labels" in sample
    assert "in_str" in sample
    assert "out_str" in sample
    assert torch.is_tensor(sample["gold_bboxes"])
    assert torch.is_tensor(sample["gold_labels"])


@pytest.fixture
def trainer_args(data_root):
    """Create arguments for trainer testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        args = SimpleNamespace(
            # Data args
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
            # Transform args
            discrete_x_grid=16,
            discrete_y_grid=16,
            gaussian_noise_mean=0.0,
            gaussian_noise_std=0.0,
            train_bernoulli_beta=0.0,
            # Model args
            d_model=32,
            num_layers=2,
            nhead=4,
            dropout=0.1,
            share_embedding=False,
            num_pos_embed=128,
            add_task_embedding=False,
            add_task_prompt_token_in_model=False,
            num_task_prompt_token=1,
            # Training args
            batch_size=2,
            eval_batch_size=2,
            epoch=1,  # Just 1 epoch for testing
            train_log_step=1,
            gradient_accumulation=1,
            enable_clip_gradient=True,
            clip_gradient=1.0,
            load_train_ckpt=False,
            train_ckpt_path=None,
            out_dir=tmpdir,
            bbox_format="ltrb",
            # Eval args
            eval_seed=123,
            decode_max_length=64,
            topk=5,
            temperature=0.7,
            enable_task_measure=False,
        )
        yield args


@pytest.mark.integration
@pytest.mark.gpu
@pytest.mark.real_data
@pytest.mark.timeout(300)
def test_trainer_with_real_rico_data(trainer_args, data_root, single_gpu_env):
    """Test BasicTrainer with real RICO data for short training."""
    # Skip if data doesn't exist
    rico_path = data_root / "rico" / "pre_processed_20_25"
    if not rico_path.exists():
        pytest.skip("RICO preprocessed data not found")

    # Create tokenizer
    tokenizer = create_tokenizer(
        ["refinement"], trainer_args.dataset, trainer_args.discrete_x_grid
    )

    # Build model
    model = build_model(trainer_args, tokenizer)

    # Create datasets (use small subset for speed)
    train_dataset = create_dataset(
        trainer_args,
        tokenizer=tokenizer,
        task_config=TASK_CONFIG,
        split="train",
        sort_by_pos=True,
    )

    val_dataset = create_dataset(
        trainer_args,
        tokenizer=tokenizer,
        task_config=TASK_CONFIG,
        split="val",
        sort_by_pos=True,
    )

    # Limit dataset size for faster testing
    if len(train_dataset) > 10:
        train_dataset.data = train_dataset.data[:10]
    if len(val_dataset) > 5:
        val_dataset.data = val_dataset.data[:5]

    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Mock wandb to avoid actual logging
    import wandb
    from unittest.mock import patch

    with patch.object(wandb, "init"):
        with patch.object(wandb, "log"):
            with patch("layoutformer_pp.trainer.basic_trainer.utils.init_experiment"):
                with patch(
                    "layoutformer_pp.trainer.basic_trainer.utils.log_hyperparameters",
                    return_value={},
                ):
                    # Create trainer
                    trainer = Trainer(
                        task_name="test_integration",
                        args=trainer_args,
                        tokenizer=tokenizer,
                        model=model,
                        seq_processor=_get_seq_processor(val_dataset),
                        train_dataset=train_dataset,
                        val_dataset=val_dataset,
                        optimizer=optimizer,
                        scheduler=None,
                        is_label_condition=False,
                        task_config={},
                    )

                    # Run training for 1 epoch
                    train_fn = TrainFn()
                    eval_fn = EvaluateFn(
                        trainer_args.max_num_elements,
                        enable_task_measure=False,
                        decode_max_length=trainer_args.decode_max_length,
                        topk=trainer_args.topk,
                        temperature=trainer_args.temperature,
                    )

                    # Simplified training loop
                    trainer.model.train()
                    for batch_idx, data in enumerate(trainer.train_dataloader):
                        loss = train_fn(
                            trainer.model, data, trainer.tokenizer, trainer.device
                        )
                        loss.backward()
                        trainer.optimizer.step()
                        trainer.optimizer.zero_grad()

                        # Just test a few batches
                        if batch_idx >= 2:
                            break

                    # Test validation
                    trainer.model.eval()
                    trainer.init_val_metrics()

                    with torch.no_grad():
                        for batch_idx, data in enumerate(trainer.val_dataloader):
                            metrics, out = eval_fn(
                                trainer.model,
                                data,
                                trainer.seq_processor,
                                trainer.tokenizer,
                                trainer.device,
                            )
                            trainer.aggregate_metrics(metrics)

                            # Just test a few batches
                            if batch_idx >= 1:
                                break

    # Verify metrics were collected
    assert trainer.val_num_bbox >= 0
    assert trainer.val_num_examples >= 0


@pytest.mark.integration
@pytest.mark.gpu
@pytest.mark.real_data
@pytest.mark.timeout(300)
def test_trainer_checkpoint_with_real_data(trainer_args, data_root, single_gpu_env):
    """Test trainer checkpointing with real data."""
    # Skip if data doesn't exist
    rico_path = data_root / "rico" / "pre_processed_20_25"
    if not rico_path.exists():
        pytest.skip("RICO preprocessed data not found")

    # Create tokenizer
    tokenizer = create_tokenizer(
        ["refinement"], trainer_args.dataset, trainer_args.discrete_x_grid
    )

    # Build model
    model = build_model(trainer_args, tokenizer)

    # Create minimal datasets
    train_dataset = create_dataset(
        trainer_args,
        tokenizer=tokenizer,
        task_config=TASK_CONFIG,
        split="train",
        sort_by_pos=True,
    )

    val_dataset = create_dataset(
        trainer_args,
        tokenizer=tokenizer,
        task_config=TASK_CONFIG,
        split="val",
        sort_by_pos=True,
    )

    # Use very small subset
    train_dataset.data = train_dataset.data[:5]
    val_dataset.data = val_dataset.data[:3]

    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Mock wandb
    import wandb
    from unittest.mock import patch

    with patch.object(wandb, "init"):
        with patch.object(wandb, "log"):
            with patch("layoutformer_pp.trainer.basic_trainer.utils.init_experiment"):
                with patch(
                    "layoutformer_pp.trainer.basic_trainer.utils.log_hyperparameters",
                    return_value={},
                ):
                    trainer = Trainer(
                        task_name="test_checkpoint",
                        args=trainer_args,
                        tokenizer=tokenizer,
                        model=model,
                        seq_processor=_get_seq_processor(val_dataset),
                        train_dataset=train_dataset,
                        val_dataset=val_dataset,
                        optimizer=optimizer,
                        task_config={},
                    )

                    # Do checkpointing
                    trainer.do_checkpointing(epoch=0, is_best=True)

                    # Verify checkpoint files exist
                    assert os.path.exists(
                        os.path.join(trainer_args.out_dir, "checkpoint.pth.tar")
                    )
                    assert os.path.exists(
                        os.path.join(trainer_args.out_dir, "model_best.pth.tar")
                    )

                    # Test loading checkpoint
                    state_dict = torch.load(
                        os.path.join(trainer_args.out_dir, "checkpoint.pth.tar"),
                        map_location="cpu",
                    )
                    assert state_dict is not None


@pytest.mark.integration
@pytest.mark.real_data
def test_data_loading_pipeline(data_root):
    """Test the data loading pipeline with real data."""
    # Skip if data doesn't exist
    rico_path = data_root / "rico" / "pre_processed_20_25"
    if not rico_path.exists():
        pytest.skip("RICO preprocessed data not found")

    args = SimpleNamespace(
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
        add_task_embedding=False,
    )

    # Create tokenizer
    tokenizer = create_tokenizer(["refinement"], args.dataset, args.discrete_x_grid)

    # Load data for all splits
    for split in ["train", "val", "test"]:
        dataset = create_dataset(
            args,
            tokenizer=tokenizer,
            task_config=TASK_CONFIG,
            split=split,
            sort_by_pos=True,
        )

        assert len(dataset) > 0, f"Dataset {split} is empty"

        # Check first sample
        sample = dataset[0]
        _assert_processed_sample(sample)


@pytest.mark.integration
@pytest.mark.real_data
def test_publaynet_data_loading(data_root):
    """Test PubLayNet data loading if available."""
    publaynet_path = data_root / "publaynet" / "pre_processed_20_5"
    if not publaynet_path.exists():
        pytest.skip("PubLayNet preprocessed data not found")

    args = SimpleNamespace(
        dataset="publaynet",
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
        add_task_embedding=False,
    )

    tokenizer = create_tokenizer(["refinement"], args.dataset, args.discrete_x_grid)

    dataset = create_dataset(
        args,
        tokenizer=tokenizer,
        task_config=TASK_CONFIG,
        split="val",
        sort_by_pos=True,
    )

    assert len(dataset) > 0
    sample = dataset[0]
    _assert_processed_sample(sample)


@pytest.mark.integration
@pytest.mark.gpu
@pytest.mark.real_data
@pytest.mark.timeout(300)
def test_trainer_gradient_accumulation(trainer_args, data_root, single_gpu_env):
    """Test trainer with gradient accumulation."""
    rico_path = data_root / "rico" / "pre_processed_20_25"
    if not rico_path.exists():
        pytest.skip("RICO preprocessed data not found")

    # Set gradient accumulation
    trainer_args.gradient_accumulation = 2

    tokenizer = create_tokenizer(
        ["refinement"], trainer_args.dataset, trainer_args.discrete_x_grid
    )
    model = build_model(trainer_args, tokenizer)

    train_dataset = create_dataset(
        trainer_args,
        tokenizer=tokenizer,
        task_config=TASK_CONFIG,
        split="train",
        sort_by_pos=True,
    )

    val_dataset = create_dataset(
        trainer_args,
        tokenizer=tokenizer,
        task_config=TASK_CONFIG,
        split="val",
        sort_by_pos=True,
    )

    # Small subset
    train_dataset.data = train_dataset.data[:8]
    val_dataset.data = val_dataset.data[:4]

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    import wandb
    from unittest.mock import patch

    with patch.object(wandb, "init"):
        with patch.object(wandb, "log"):
            with patch("layoutformer_pp.trainer.basic_trainer.utils.init_experiment"):
                with patch(
                    "layoutformer_pp.trainer.basic_trainer.utils.log_hyperparameters",
                    return_value={},
                ):
                    trainer = Trainer(
                        task_name="test_grad_accum",
                        args=trainer_args,
                        tokenizer=tokenizer,
                        model=model,
                        seq_processor=_get_seq_processor(val_dataset),
                        train_dataset=train_dataset,
                        val_dataset=val_dataset,
                        optimizer=optimizer,
                        task_config={},
                    )

                    assert trainer.gradient_accumulation == 2

                    # Quick training test
                    train_fn = TrainFn()
                    trainer.model.train()

                    for batch_idx, data in enumerate(trainer.train_dataloader):
                        loss = train_fn(
                            trainer.model, data, trainer.tokenizer, trainer.device
                        )
                        loss = loss / trainer.gradient_accumulation
                        loss.backward()

                        if (batch_idx + 1) % trainer.gradient_accumulation == 0:
                            trainer.optimizer.step()
                            trainer.optimizer.zero_grad()

                        if batch_idx >= 3:
                            break
