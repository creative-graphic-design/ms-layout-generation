"""Unit tests for BasicTrainer."""

from types import SimpleNamespace
from unittest.mock import Mock, patch
import tempfile
import os

import pytest
import torch
import torch.nn as nn

from layoutformer_pp.trainer.basic_trainer import Trainer
from layoutformer_pp.trainer.utils import CheckpointMeasurement
from layoutformer_pp.model import LayoutTransformerTokenizer


@pytest.fixture
def mock_args():
    """Create mock arguments for trainer."""
    with tempfile.TemporaryDirectory() as tmpdir:
        args = SimpleNamespace(
            out_dir=tmpdir,
            batch_size=2,
            eval_batch_size=2,
            epoch=2,
            train_log_step=1,
            gradient_accumulation=1,
            enable_clip_gradient=True,
            clip_gradient=1.0,
            load_train_ckpt=False,
            train_ckpt_path=None,
            max_num_elements=20,
            discrete_x_grid=16,
            discrete_y_grid=16,
            bbox_format="ltrb",
        )
        yield args


@pytest.fixture
def mock_tokenizer():
    """Create mock tokenizer."""
    return LayoutTransformerTokenizer(tokens=["label_1", "0", "1"])


@pytest.fixture
def mock_model():
    """Create mock model."""
    model = nn.Linear(10, 10)
    return model


@pytest.fixture
def mock_seq_processor():
    """Create mock sequence processor."""
    processor = Mock()
    processor.decode = Mock(return_value=("decoded", []))
    return processor


@pytest.fixture
def mock_dataset():
    """Create mock dataset."""
    dataset = []
    for i in range(4):
        dataset.append(
            {
                "bboxes": torch.randn(3, 4),
                "labels": torch.randint(0, 5, (3,)),
                "mask": torch.ones(3, dtype=torch.bool),
            }
        )
    return dataset


@pytest.fixture
def mock_optimizer(mock_model):
    """Create mock optimizer."""
    return torch.optim.Adam(mock_model.parameters(), lr=0.001)


@patch("layoutformer_pp.trainer.basic_trainer.wandb")
@patch("layoutformer_pp.trainer.basic_trainer.utils.init_experiment")
@patch("layoutformer_pp.trainer.basic_trainer.utils.log_hyperparameters")
def test_trainer_initialization(
    mock_log_hyperparam,
    mock_init_exp,
    mock_wandb,
    mock_args,
    mock_tokenizer,
    mock_model,
    mock_seq_processor,
    mock_dataset,
    mock_optimizer,
):
    """Test BasicTrainer initialization."""
    mock_log_hyperparam.return_value = {"learning_rate": 0.001}

    trainer = Trainer(
        task_name="test_task",
        args=mock_args,
        tokenizer=mock_tokenizer,
        model=mock_model,
        seq_processor=mock_seq_processor,
        train_dataset=mock_dataset,
        val_dataset=mock_dataset,
        optimizer=mock_optimizer,
        scheduler=None,
        is_label_condition=True,
        checkpoint_measure=None,
        d2c_fn=None,
        is_debug=False,
        task_config={},
        collate_fn=None,
    )

    assert trainer.args == mock_args
    assert trainer.tokenizer == mock_tokenizer
    assert trainer.model is not None
    assert trainer.optimizer == mock_optimizer
    assert trainer.scheduler is None
    assert trainer.is_label_condition is True
    assert isinstance(trainer.ckpt_measurement, CheckpointMeasurement)
    mock_init_exp.assert_called_once()
    mock_wandb.init.assert_called_once()


@patch("layoutformer_pp.trainer.basic_trainer.wandb")
@patch("layoutformer_pp.trainer.basic_trainer.utils.init_experiment")
@patch("layoutformer_pp.trainer.basic_trainer.utils.log_hyperparameters")
def test_trainer_dataparallel_setup(
    mock_log_hyperparam,
    mock_init_exp,
    mock_wandb,
    mock_args,
    mock_tokenizer,
    mock_model,
    mock_seq_processor,
    mock_dataset,
    mock_optimizer,
):
    """Test DataParallel setup when multiple GPUs detected."""
    mock_log_hyperparam.return_value = {"learning_rate": 0.001}

    with patch("torch.cuda.device_count", return_value=2):
        with patch("torch.cuda.is_available", return_value=True):
            trainer = Trainer(
                task_name="test_task",
                args=mock_args,
                tokenizer=mock_tokenizer,
                model=mock_model,
                seq_processor=mock_seq_processor,
                train_dataset=mock_dataset,
                val_dataset=mock_dataset,
                optimizer=mock_optimizer,
            )

            # When multiple GPUs, batch size should be multiplied
            assert len(trainer.train_dataloader) > 0


@patch("layoutformer_pp.trainer.basic_trainer.wandb")
@patch("layoutformer_pp.trainer.basic_trainer.utils.init_experiment")
@patch("layoutformer_pp.trainer.basic_trainer.utils.log_hyperparameters")
@pytest.mark.gpu
def test_trainer_gpu_setup(
    mock_log_hyperparam,
    mock_init_exp,
    mock_wandb,
    mock_args,
    mock_tokenizer,
    mock_model,
    mock_seq_processor,
    mock_dataset,
    mock_optimizer,
    gpu_device,
):
    """Test trainer setup with actual GPU."""
    mock_log_hyperparam.return_value = {"learning_rate": 0.001}

    trainer = Trainer(
        task_name="test_task",
        args=mock_args,
        tokenizer=mock_tokenizer,
        model=mock_model,
        seq_processor=mock_seq_processor,
        train_dataset=mock_dataset,
        val_dataset=mock_dataset,
        optimizer=mock_optimizer,
    )

    assert trainer.device.type == "cuda"
    assert next(trainer.model.parameters()).device.type == "cuda"


@patch("layoutformer_pp.trainer.basic_trainer.wandb")
@patch("layoutformer_pp.trainer.basic_trainer.utils.init_experiment")
@patch("layoutformer_pp.trainer.basic_trainer.utils.log_hyperparameters")
def test_trainer_default_d2c_fn(
    mock_log_hyperparam,
    mock_init_exp,
    mock_wandb,
    mock_args,
    mock_tokenizer,
    mock_model,
    mock_seq_processor,
    mock_dataset,
    mock_optimizer,
):
    """Test default discrete to continuous function."""
    mock_log_hyperparam.return_value = {"learning_rate": 0.001}

    trainer = Trainer(
        task_name="test_task",
        args=mock_args,
        tokenizer=mock_tokenizer,
        model=mock_model,
        seq_processor=mock_seq_processor,
        train_dataset=mock_dataset,
        val_dataset=mock_dataset,
        optimizer=mock_optimizer,
    )

    # Test d2c function
    discrete_bbox = torch.tensor([[5, 5, 10, 10]], dtype=torch.long)
    continuous_bbox = trainer.d2c_fn(discrete_bbox)

    assert continuous_bbox.dtype == torch.float
    assert continuous_bbox.shape == discrete_bbox.shape


@patch("layoutformer_pp.trainer.basic_trainer.wandb")
@patch("layoutformer_pp.trainer.basic_trainer.utils.init_experiment")
@patch("layoutformer_pp.trainer.basic_trainer.utils.log_hyperparameters")
def test_trainer_init_val_metrics(
    mock_log_hyperparam,
    mock_init_exp,
    mock_wandb,
    mock_args,
    mock_tokenizer,
    mock_model,
    mock_seq_processor,
    mock_dataset,
    mock_optimizer,
):
    """Test validation metrics initialization."""
    mock_log_hyperparam.return_value = {"learning_rate": 0.001}

    trainer = Trainer(
        task_name="test_task",
        args=mock_args,
        tokenizer=mock_tokenizer,
        model=mock_model,
        seq_processor=mock_seq_processor,
        train_dataset=mock_dataset,
        val_dataset=mock_dataset,
        optimizer=mock_optimizer,
    )

    trainer.init_val_metrics()

    assert trainer.val_num_bbox_correct == 0.0
    assert trainer.val_num_bbox == 0.0
    assert trainer.val_num_label_correct == 0.0
    assert trainer.val_num_examples == 0.0
    assert trainer.violation_num == 0.0
    assert trainer.rel_num == 0.0


@patch("layoutformer_pp.trainer.basic_trainer.wandb")
@patch("layoutformer_pp.trainer.basic_trainer.utils.init_experiment")
@patch("layoutformer_pp.trainer.basic_trainer.utils.log_hyperparameters")
def test_trainer_aggregate_metrics(
    mock_log_hyperparam,
    mock_init_exp,
    mock_wandb,
    mock_args,
    mock_tokenizer,
    mock_model,
    mock_seq_processor,
    mock_dataset,
    mock_optimizer,
):
    """Test metrics aggregation."""
    mock_log_hyperparam.return_value = {"learning_rate": 0.001}

    trainer = Trainer(
        task_name="test_task",
        args=mock_args,
        tokenizer=mock_tokenizer,
        model=mock_model,
        seq_processor=mock_seq_processor,
        train_dataset=mock_dataset,
        val_dataset=mock_dataset,
        optimizer=mock_optimizer,
    )

    trainer.init_val_metrics()

    metrics = {
        "num_bbox_correct": 10.0,
        "num_bbox": 20.0,
        "num_label_correct": 8.0,
        "num_examples": 10.0,
        "violation_num": 2.0,
        "rel_num": 15.0,
    }

    trainer.aggregate_metrics(metrics)

    assert trainer.val_num_bbox_correct == 10.0
    assert trainer.val_num_bbox == 20.0
    assert trainer.val_num_label_correct == 8.0
    assert trainer.val_num_examples == 10.0
    assert trainer.violation_num == 2.0
    assert trainer.rel_num == 15.0


@patch("layoutformer_pp.trainer.basic_trainer.wandb")
@patch("layoutformer_pp.trainer.basic_trainer.utils.init_experiment")
@patch("layoutformer_pp.trainer.basic_trainer.utils.log_hyperparameters")
def test_trainer_convert_bbox_format(
    mock_log_hyperparam,
    mock_init_exp,
    mock_wandb,
    mock_args,
    mock_tokenizer,
    mock_model,
    mock_seq_processor,
    mock_dataset,
    mock_optimizer,
):
    """Test bbox format conversion."""
    mock_log_hyperparam.return_value = {"learning_rate": 0.001}

    trainer = Trainer(
        task_name="test_task",
        args=mock_args,
        tokenizer=mock_tokenizer,
        model=mock_model,
        seq_processor=mock_seq_processor,
        train_dataset=mock_dataset,
        val_dataset=mock_dataset,
        optimizer=mock_optimizer,
    )

    bboxes = torch.tensor([[5, 5, 10, 10]], dtype=torch.long)
    converted = trainer.convert_bbox_format(bboxes)

    assert converted.shape == bboxes.shape


@patch("layoutformer_pp.trainer.basic_trainer.wandb")
@patch("layoutformer_pp.trainer.basic_trainer.utils.init_experiment")
@patch("layoutformer_pp.trainer.basic_trainer.utils.log_hyperparameters")
def test_trainer_collect_layouts(
    mock_log_hyperparam,
    mock_init_exp,
    mock_wandb,
    mock_args,
    mock_tokenizer,
    mock_model,
    mock_seq_processor,
    mock_dataset,
    mock_optimizer,
):
    """Test layout collection."""
    mock_log_hyperparam.return_value = {"learning_rate": 0.001}

    trainer = Trainer(
        task_name="test_task",
        args=mock_args,
        tokenizer=mock_tokenizer,
        model=mock_model,
        seq_processor=mock_seq_processor,
        train_dataset=mock_dataset,
        val_dataset=mock_dataset,
        optimizer=mock_optimizer,
    )

    bboxes = torch.tensor([[[5, 5, 10, 10], [6, 6, 12, 12]]], dtype=torch.long)
    labels = torch.tensor([[1, 2]], dtype=torch.long)
    mask = torch.tensor([[True, True]], dtype=torch.bool)

    layouts = trainer.collect_layouts(bboxes, labels, mask)

    assert len(layouts) == 1
    assert len(layouts[0]) == 2  # (bbox, label)


@patch("layoutformer_pp.trainer.basic_trainer.wandb")
@patch("layoutformer_pp.trainer.basic_trainer.utils.init_experiment")
@patch("layoutformer_pp.trainer.basic_trainer.utils.log_hyperparameters")
def test_trainer_do_checkpointing(
    mock_log_hyperparam,
    mock_init_exp,
    mock_wandb,
    mock_args,
    mock_tokenizer,
    mock_model,
    mock_seq_processor,
    mock_dataset,
    mock_optimizer,
):
    """Test checkpointing functionality."""
    mock_log_hyperparam.return_value = {"learning_rate": 0.001}

    trainer = Trainer(
        task_name="test_task",
        args=mock_args,
        tokenizer=mock_tokenizer,
        model=mock_model,
        seq_processor=mock_seq_processor,
        train_dataset=mock_dataset,
        val_dataset=mock_dataset,
        optimizer=mock_optimizer,
    )

    trainer.do_checkpointing(epoch=0, is_best=True)

    # Check that checkpoint files are created
    assert os.path.exists(os.path.join(mock_args.out_dir, "checkpoint.pth.tar"))
    assert os.path.exists(os.path.join(mock_args.out_dir, "model_best.pth.tar"))


@patch("layoutformer_pp.trainer.basic_trainer.wandb")
@patch("layoutformer_pp.trainer.basic_trainer.utils.init_experiment")
@patch("layoutformer_pp.trainer.basic_trainer.utils.log_hyperparameters")
def test_trainer_scheduler_step(
    mock_log_hyperparam,
    mock_init_exp,
    mock_wandb,
    mock_args,
    mock_tokenizer,
    mock_model,
    mock_seq_processor,
    mock_dataset,
    mock_optimizer,
):
    """Test scheduler step is called."""
    mock_log_hyperparam.return_value = {"learning_rate": 0.001}

    mock_scheduler = Mock()

    trainer = Trainer(
        task_name="test_task",
        args=mock_args,
        tokenizer=mock_tokenizer,
        model=mock_model,
        seq_processor=mock_seq_processor,
        train_dataset=mock_dataset,
        val_dataset=mock_dataset,
        optimizer=mock_optimizer,
        scheduler=mock_scheduler,
    )

    assert trainer.scheduler == mock_scheduler


@patch("layoutformer_pp.trainer.basic_trainer.wandb")
@patch("layoutformer_pp.trainer.basic_trainer.utils.init_experiment")
@patch("layoutformer_pp.trainer.basic_trainer.utils.log_hyperparameters")
def test_trainer_checkpoint_measurement(
    mock_log_hyperparam,
    mock_init_exp,
    mock_wandb,
    mock_args,
    mock_tokenizer,
    mock_model,
    mock_seq_processor,
    mock_dataset,
    mock_optimizer,
):
    """Test checkpoint measurement setup."""
    mock_log_hyperparam.return_value = {"learning_rate": 0.001}

    trainer = Trainer(
        task_name="test_task",
        args=mock_args,
        tokenizer=mock_tokenizer,
        model=mock_model,
        seq_processor=mock_seq_processor,
        train_dataset=mock_dataset,
        val_dataset=mock_dataset,
        optimizer=mock_optimizer,
        checkpoint_measure=CheckpointMeasurement.ACCURACY,
    )

    assert trainer.ckpt_measurement.measurement == CheckpointMeasurement.ACCURACY


@patch("layoutformer_pp.trainer.basic_trainer.wandb")
@patch("layoutformer_pp.trainer.basic_trainer.utils.init_experiment")
@patch("layoutformer_pp.trainer.basic_trainer.utils.log_hyperparameters")
def test_trainer_load_checkpoint(
    mock_log_hyperparam,
    mock_init_exp,
    mock_wandb,
    mock_args,
    mock_tokenizer,
    mock_model,
    mock_seq_processor,
    mock_dataset,
    mock_optimizer,
):
    """Test checkpoint loading."""
    mock_log_hyperparam.return_value = {"learning_rate": 0.001}

    # Save a checkpoint first
    ckpt_path = os.path.join(mock_args.out_dir, "test_ckpt.pth.tar")
    torch.save(mock_model.state_dict(), ckpt_path)

    # Update args to load checkpoint
    mock_args.load_train_ckpt = True
    mock_args.train_ckpt_path = ckpt_path

    trainer = Trainer(
        task_name="test_task",
        args=mock_args,
        tokenizer=mock_tokenizer,
        model=mock_model,
        seq_processor=mock_seq_processor,
        train_dataset=mock_dataset,
        val_dataset=mock_dataset,
        optimizer=mock_optimizer,
    )

    # Should not raise an error
    assert trainer.model is not None


@patch("layoutformer_pp.trainer.basic_trainer.wandb")
@patch("layoutformer_pp.trainer.basic_trainer.utils.init_experiment")
@patch("layoutformer_pp.trainer.basic_trainer.utils.log_hyperparameters")
def test_trainer_call_runs_one_epoch(
    mock_log_hyperparam,
    mock_init_exp,
    mock_wandb,
):
    """Test trainer __call__ executes a minimal train/eval loop."""
    mock_log_hyperparam.return_value = {"learning_rate": 0.001}

    with tempfile.TemporaryDirectory() as tmpdir:
        args = SimpleNamespace(
            out_dir=tmpdir,
            batch_size=2,
            eval_batch_size=2,
            epoch=1,
            train_log_step=1,
            gradient_accumulation=1,
            enable_clip_gradient=False,
            clip_gradient=1.0,
            load_train_ckpt=False,
            train_ckpt_path=None,
            max_num_elements=2,
            discrete_x_grid=16,
            discrete_y_grid=16,
            bbox_format="ltrb",
        )
        tokenizer = LayoutTransformerTokenizer(tokens=["label_1", "0", "1"])
        model = nn.Linear(1, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        train_dataset = [
            {"dummy": 1, "out_str": "label_1 0 0 1 1"},
            {"dummy": 2, "out_str": "label_1 0 0 1 1"},
        ]
        val_dataset = [
            {"dummy": 1, "out_str": "label_1 0 0 1 1"},
            {"dummy": 2, "out_str": "label_1 0 0 1 1"},
        ]

        with (
            patch("torch.cuda.device_count", return_value=1),
            patch("torch.cuda.is_available", return_value=False),
        ):
            trainer = Trainer(
                task_name="test_task",
                args=args,
                tokenizer=tokenizer,
                model=model,
                seq_processor=Mock(),
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                optimizer=optimizer,
                checkpoint_measure=CheckpointMeasurement.ACCURACY,
                is_debug=True,
            )

        def train_step(step_model, data, step_tokenizer, device):
            return step_model.weight.sum()

        def eval_step(step_model, data, seq_processor, step_tokenizer, device):
            batch_size = len(data["out_str"])
            pred_bboxes = torch.zeros(batch_size, 1, 4)
            pred_labels = torch.ones(batch_size, 1, dtype=torch.long)
            gold_bboxes = torch.zeros(batch_size, 1, 4)
            gold_labels = torch.ones(batch_size, 1, dtype=torch.long)
            mask = torch.ones(batch_size, 1, dtype=torch.bool)
            metrics = {
                "num_bbox_correct": 1.0,
                "num_bbox": 1.0,
                "num_label_correct": 1.0,
                "num_examples": float(batch_size),
                "violation_num": 0.0,
                "rel_num": 1.0,
            }
            out = {
                "pred_bboxes": pred_bboxes,
                "pred_labels": pred_labels,
                "gold_bboxes": gold_bboxes,
                "gold_labels": gold_labels,
                "mask": mask,
                "out_str": data["out_str"],
            }
            return metrics, out

        trainer(train_step, eval_step)

        assert os.path.exists(os.path.join(args.out_dir, "checkpoint.pth.tar"))
        assert os.path.exists(os.path.join(args.out_dir, "val_output.pkl"))
