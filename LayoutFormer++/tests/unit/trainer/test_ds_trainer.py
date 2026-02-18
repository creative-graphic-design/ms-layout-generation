"""Unit tests for DSTrainer."""

import os
import tempfile
from types import SimpleNamespace
from unittest.mock import Mock, patch, MagicMock

import pytest
import torch
import torch.nn as nn

from layoutformer_pp.trainer.ds_trainer import DSTrainer
from layoutformer_pp.model import LayoutTransformerTokenizer


@pytest.fixture
def mock_args():
    """Create mock arguments for DS trainer."""
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
            ds_ckpt_tag="latest",
            max_num_elements=20,
            discrete_x_grid=16,
            discrete_y_grid=16,
            bbox_format="ltrb",
            backend="nccl",
            deepscale_config=os.path.join(tmpdir, "ds_config.json"),
        )
        yield args


@pytest.fixture
def mock_ds_config(mock_args):
    """Create mock DeepSpeed config."""
    ds_config = {
        "train_micro_batch_size_per_gpu": 2,
        "gradient_accumulation_steps": 1,
        "scheduler": {
            "params": {
                "warmup_num_steps": 100,
            }
        },
    }
    # Write config to file
    import json

    with open(mock_args.deepscale_config, "w") as f:
        json.dump(ds_config, f)
    return ds_config


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
def mock_optimizer():
    """Create mock optimizer."""
    return Mock()


@pytest.fixture
def mock_deepspeed():
    """Create mock DeepSpeed module."""
    with patch("layoutformer_pp.trainer.ds_trainer.deepspeed") as mock_ds:
        mock_engine = MagicMock()
        mock_engine.local_rank = 0
        mock_engine.backward = Mock()
        mock_engine.step = Mock()
        mock_engine.train = Mock()
        mock_engine.eval = Mock()
        mock_engine.save_checkpoint = Mock()
        mock_engine.load_checkpoint = Mock(return_value=(None, {}))

        mock_ds.initialize = Mock(return_value=(mock_engine, Mock(), None, None))
        mock_ds.init_distributed = Mock()

        yield mock_ds


@patch.dict(os.environ, {"LOCAL_RANK": "0", "WORLD_SIZE": "1"}, clear=False)
@patch("layoutformer_pp.trainer.ds_trainer.wandb")
@patch("layoutformer_pp.trainer.ds_trainer.utils.init_experiment")
@patch("layoutformer_pp.trainer.ds_trainer.utils.log_hyperparameters")
@patch("layoutformer_pp.trainer.ds_trainer.utils.load_arguments")
@patch("torch.distributed.barrier")
def test_ds_trainer_initialization(
    mock_barrier,
    mock_load_args,
    mock_log_hyperparam,
    mock_init_exp,
    mock_wandb,
    mock_deepspeed,
    mock_ds_config,
    mock_args,
    mock_tokenizer,
    mock_model,
    mock_seq_processor,
    mock_dataset,
    mock_optimizer,
):
    """Test DSTrainer initialization."""
    mock_load_args.return_value = mock_ds_config
    mock_log_hyperparam.return_value = {"learning_rate": 0.001}

    trainer = DSTrainer(
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

    assert trainer._local_rank == 0
    assert trainer._world_size == 1
    mock_deepspeed.init_distributed.assert_called_once()


@patch.dict(os.environ, {"LOCAL_RANK": "0", "WORLD_SIZE": "1"}, clear=False)
@patch("layoutformer_pp.trainer.ds_trainer.wandb")
@patch("layoutformer_pp.trainer.ds_trainer.utils.init_experiment")
@patch("layoutformer_pp.trainer.ds_trainer.utils.log_hyperparameters")
@patch("layoutformer_pp.trainer.ds_trainer.utils.load_arguments")
@patch("torch.distributed.barrier")
def test_ds_trainer_is_main_process(
    mock_barrier,
    mock_load_args,
    mock_log_hyperparam,
    mock_init_exp,
    mock_wandb,
    mock_deepspeed,
    mock_ds_config,
    mock_args,
    mock_tokenizer,
    mock_model,
    mock_seq_processor,
    mock_dataset,
    mock_optimizer,
):
    """Test main process detection."""
    mock_load_args.return_value = mock_ds_config
    mock_log_hyperparam.return_value = {"learning_rate": 0.001}

    trainer = DSTrainer(
        task_name="test_task",
        args=mock_args,
        tokenizer=mock_tokenizer,
        model=mock_model,
        seq_processor=mock_seq_processor,
        train_dataset=mock_dataset,
        val_dataset=mock_dataset,
        optimizer=mock_optimizer,
    )

    assert trainer._is_main_process is True


@patch.dict(os.environ, {"LOCAL_RANK": "0", "WORLD_SIZE": "1"}, clear=False)
@patch("layoutformer_pp.trainer.ds_trainer.wandb")
@patch("layoutformer_pp.trainer.ds_trainer.utils.init_experiment")
@patch("layoutformer_pp.trainer.ds_trainer.utils.log_hyperparameters")
@patch("layoutformer_pp.trainer.ds_trainer.utils.load_arguments")
@patch("torch.distributed.barrier")
def test_ds_trainer_dataloader_setup(
    mock_barrier,
    mock_load_args,
    mock_log_hyperparam,
    mock_init_exp,
    mock_wandb,
    mock_deepspeed,
    mock_ds_config,
    mock_args,
    mock_tokenizer,
    mock_model,
    mock_seq_processor,
    mock_dataset,
    mock_optimizer,
):
    """Test DeepSpeed dataloader setup with DistributedSampler."""
    mock_load_args.return_value = mock_ds_config
    mock_log_hyperparam.return_value = {"learning_rate": 0.001}

    trainer = DSTrainer(
        task_name="test_task",
        args=mock_args,
        tokenizer=mock_tokenizer,
        model=mock_model,
        seq_processor=mock_seq_processor,
        train_dataset=mock_dataset,
        val_dataset=mock_dataset,
        optimizer=mock_optimizer,
    )

    assert trainer.train_dataloader is not None
    assert trainer.val_dataloader is not None
    assert hasattr(trainer, "train_sampler")


@patch.dict(os.environ, {"LOCAL_RANK": "0", "WORLD_SIZE": "1"}, clear=False)
@patch("layoutformer_pp.trainer.ds_trainer.wandb")
@patch("layoutformer_pp.trainer.ds_trainer.utils.init_experiment")
@patch("layoutformer_pp.trainer.ds_trainer.utils.log_hyperparameters")
@patch("layoutformer_pp.trainer.ds_trainer.utils.load_arguments")
@patch("torch.distributed.barrier")
def test_ds_trainer_config_assertion(
    mock_barrier,
    mock_load_args,
    mock_log_hyperparam,
    mock_init_exp,
    mock_wandb,
    mock_deepspeed,
    mock_args,
    mock_tokenizer,
    mock_model,
    mock_seq_processor,
    mock_dataset,
    mock_optimizer,
):
    """Test config validation assertions."""
    # Config with mismatched batch size
    wrong_config = {
        "train_micro_batch_size_per_gpu": 999,  # Different from args
        "gradient_accumulation_steps": 1,
        "scheduler": {
            "params": {
                "warmup_num_steps": 100,
            }
        },
    }
    mock_load_args.return_value = wrong_config
    mock_log_hyperparam.return_value = {"learning_rate": 0.001}

    with pytest.raises(AssertionError):
        DSTrainer(
            task_name="test_task",
            args=mock_args,
            tokenizer=mock_tokenizer,
            model=mock_model,
            seq_processor=mock_seq_processor,
            train_dataset=mock_dataset,
            val_dataset=mock_dataset,
            optimizer=mock_optimizer,
        )


@patch.dict(os.environ, {"LOCAL_RANK": "0", "WORLD_SIZE": "1"}, clear=False)
@patch("layoutformer_pp.trainer.ds_trainer.wandb")
@patch("layoutformer_pp.trainer.ds_trainer.utils.init_experiment")
@patch("layoutformer_pp.trainer.ds_trainer.utils.log_hyperparameters")
@patch("layoutformer_pp.trainer.ds_trainer.utils.load_arguments")
@patch("torch.distributed.barrier")
def test_ds_trainer_checkpoint_save(
    mock_barrier,
    mock_load_args,
    mock_log_hyperparam,
    mock_init_exp,
    mock_wandb,
    mock_deepspeed,
    mock_ds_config,
    mock_args,
    mock_tokenizer,
    mock_model,
    mock_seq_processor,
    mock_dataset,
    mock_optimizer,
):
    """Test DeepSpeed checkpoint saving."""
    mock_load_args.return_value = mock_ds_config
    mock_log_hyperparam.return_value = {"learning_rate": 0.001}

    trainer = DSTrainer(
        task_name="test_task",
        args=mock_args,
        tokenizer=mock_tokenizer,
        model=mock_model,
        seq_processor=mock_seq_processor,
        train_dataset=mock_dataset,
        val_dataset=mock_dataset,
        optimizer=mock_optimizer,
    )

    # Test checkpointing
    trainer.do_checkpointing(epoch=0, is_best=True)

    # Verify checkpoint directory handling
    assert hasattr(trainer, "_normal_ckpt_path")


@patch.dict(os.environ, {"LOCAL_RANK": "1", "WORLD_SIZE": "2"}, clear=False)
@patch("layoutformer_pp.trainer.ds_trainer.wandb")
@patch("layoutformer_pp.trainer.ds_trainer.utils.init_experiment")
@patch("layoutformer_pp.trainer.ds_trainer.utils.log_hyperparameters")
@patch("layoutformer_pp.trainer.ds_trainer.utils.load_arguments")
@patch("torch.distributed.barrier")
def test_ds_trainer_non_main_process(
    mock_barrier,
    mock_load_args,
    mock_log_hyperparam,
    mock_init_exp,
    mock_wandb,
    mock_deepspeed,
    mock_ds_config,
    mock_args,
    mock_tokenizer,
    mock_model,
    mock_seq_processor,
    mock_dataset,
    mock_optimizer,
):
    """Test non-main process doesn't initialize wandb."""
    mock_load_args.return_value = mock_ds_config
    mock_log_hyperparam.return_value = {"learning_rate": 0.001}

    trainer = DSTrainer(
        task_name="test_task",
        args=mock_args,
        tokenizer=mock_tokenizer,
        model=mock_model,
        seq_processor=mock_seq_processor,
        train_dataset=mock_dataset,
        val_dataset=mock_dataset,
        optimizer=mock_optimizer,
    )

    assert trainer._is_main_process is False
    # wandb.init should not be called for non-main process
    # It's only called if _is_main_process is True in _setup_experiment


@patch.dict(os.environ, {"LOCAL_RANK": "0", "WORLD_SIZE": "1"}, clear=False)
@patch("layoutformer_pp.trainer.ds_trainer.wandb")
@patch("layoutformer_pp.trainer.ds_trainer.utils.init_experiment")
@patch("layoutformer_pp.trainer.ds_trainer.utils.log_hyperparameters")
@patch("layoutformer_pp.trainer.ds_trainer.utils.load_arguments")
@patch("torch.distributed.barrier")
def test_ds_trainer_load_checkpoint(
    mock_barrier,
    mock_load_args,
    mock_log_hyperparam,
    mock_init_exp,
    mock_wandb,
    mock_deepspeed,
    mock_ds_config,
    mock_args,
    mock_tokenizer,
    mock_model,
    mock_seq_processor,
    mock_dataset,
    mock_optimizer,
):
    """Test checkpoint loading in DeepSpeed."""
    mock_load_args.return_value = mock_ds_config
    mock_log_hyperparam.return_value = {"learning_rate": 0.001}

    # Set up checkpoint loading
    mock_args.load_train_ckpt = True
    mock_args.train_ckpt_path = "/fake/path"

    trainer = DSTrainer(
        task_name="test_task",
        args=mock_args,
        tokenizer=mock_tokenizer,
        model=mock_model,
        seq_processor=mock_seq_processor,
        train_dataset=mock_dataset,
        val_dataset=mock_dataset,
        optimizer=mock_optimizer,
    )

    # Verify load_checkpoint was called
    trainer.model_engine.load_checkpoint.assert_called_once()


@patch.dict(os.environ, {"LOCAL_RANK": "0", "WORLD_SIZE": "1"}, clear=False)
@patch("layoutformer_pp.trainer.ds_trainer.wandb")
@patch("layoutformer_pp.trainer.ds_trainer.utils.init_experiment")
@patch("layoutformer_pp.trainer.ds_trainer.utils.log_hyperparameters")
@patch("layoutformer_pp.trainer.ds_trainer.utils.load_arguments")
@patch("torch.distributed.barrier")
def test_ds_trainer_call_runs_one_epoch(
    mock_barrier,
    mock_load_args,
    mock_log_hyperparam,
    mock_init_exp,
    mock_wandb,
    mock_deepspeed,
    mock_ds_config,
    mock_args,
    mock_tokenizer,
    mock_model,
    mock_seq_processor,
    mock_dataset,
    mock_optimizer,
):
    mock_load_args.return_value = mock_ds_config
    mock_log_hyperparam.return_value = {"learning_rate": 0.001}
    mock_args.epoch = 1

    custom_dataset = [
        {"mask": torch.ones(1, dtype=torch.bool), "out_str": "out"} for _ in range(4)
    ]

    trainer = DSTrainer(
        task_name="test_task",
        args=mock_args,
        tokenizer=mock_tokenizer,
        model=mock_model,
        seq_processor=mock_seq_processor,
        train_dataset=custom_dataset,
        val_dataset=custom_dataset,
        optimizer=mock_optimizer,
        is_debug=True,
    )

    trainer.ckpt_measurement.compute = Mock(return_value=0.0)
    trainer.ckpt_measurement.update = Mock(return_value=True)

    os.makedirs(trainer._normal_ckpt_path, exist_ok=True)

    def train_step(model_engine, data, tokenizer, device):
        return torch.tensor(1.0)

    def eval_step(model_engine, data, seq_processor, tokenizer, device):
        batch_size = len(data["mask"])
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
            "out_str": ["out"] * batch_size,
        }
        return metrics, out

    trainer(train_step, eval_step)

    best_ckpt_path = os.path.join(mock_args.out_dir, "model_best.pth.tar")
    assert os.path.exists(best_ckpt_path)
