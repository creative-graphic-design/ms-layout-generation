"""Unit tests for MultiTaskTrainer and MultiTaskBatchSampler."""

import os
import tempfile
from types import SimpleNamespace
from unittest.mock import Mock, patch, MagicMock

import pytest
import torch
import torch.nn as nn
import numpy as np

from layoutformer_pp.trainer.multitask_trainer import (
    MultiTaskBatchSampler,
    MultiTaskTrainer,
    DSMultiTaskTrainer,
)
from layoutformer_pp.model import LayoutTransformerTokenizer
from torch.utils.data import RandomSampler


@pytest.fixture
def mock_sampler():
    """Create mock sampler."""
    return RandomSampler(list(range(100)))


def test_multitask_batch_sampler_initialization(mock_sampler):
    """Test MultiTaskBatchSampler initialization."""
    sampler = MultiTaskBatchSampler(
        sampler=mock_sampler,
        batch_size=4,
        drop_last=True,
        num_tasks=3,
        task_sample_weights=None,
    )

    assert sampler._num_tasks == 3
    assert len(sampler._task_sample_probs) == 3
    assert np.allclose(sampler._task_sample_probs, [1 / 3, 1 / 3, 1 / 3])


def test_multitask_batch_sampler_with_weights(mock_sampler):
    """Test MultiTaskBatchSampler with custom weights."""
    sampler = MultiTaskBatchSampler(
        sampler=mock_sampler,
        batch_size=4,
        drop_last=True,
        num_tasks=3,
        task_sample_weights="1.0,2.0,3.0",
    )

    expected_probs = np.array([1.0, 2.0, 3.0])
    expected_probs = expected_probs / expected_probs.sum()

    assert np.allclose(sampler._task_sample_probs, expected_probs)


def test_multitask_batch_sampler_weights_mismatch(mock_sampler):
    """Test error when weights don't match number of tasks."""
    with pytest.raises(ValueError, match="Task sample weights should be equivalent"):
        MultiTaskBatchSampler(
            sampler=mock_sampler,
            batch_size=4,
            drop_last=True,
            num_tasks=3,
            task_sample_weights="1.0,2.0",  # Only 2 weights for 3 tasks
        )


def test_multitask_batch_sampler_sample_task(mock_sampler):
    """Test task sampling."""
    sampler = MultiTaskBatchSampler(
        sampler=mock_sampler,
        batch_size=4,
        drop_last=True,
        num_tasks=3,
        task_sample_weights=None,
    )

    # Sample multiple times to check distribution
    np.random.seed(42)
    samples = [sampler._sample_task() for _ in range(100)]

    # All samples should be valid task IDs
    assert all(0 <= s < 3 for s in samples)
    # Should have some variety (not all the same)
    assert len(set(samples)) > 1


def test_multitask_batch_sampler_iteration(mock_sampler):
    """Test batch sampling iteration."""
    sampler = MultiTaskBatchSampler(
        sampler=RandomSampler(list(range(20))),
        batch_size=4,
        drop_last=True,
        num_tasks=2,
        task_sample_weights=None,
    )

    batches = list(sampler)

    # Check that batches are created
    assert len(batches) > 0

    # Check batch size
    for batch in batches:
        assert len(batch) == 4

    # Check that indices are in valid range
    for batch in batches:
        for idx in batch:
            assert 0 <= idx < 40  # 2 tasks * 20 samples


def test_multitask_batch_sampler_drop_last_false():
    """Test batch sampling without dropping last incomplete batch."""
    sampler = MultiTaskBatchSampler(
        sampler=RandomSampler(list(range(10))),
        batch_size=4,
        drop_last=False,
        num_tasks=2,
        task_sample_weights=None,
    )

    batches = list(sampler)

    # Should include incomplete batch at end
    total_samples = sum(len(batch) for batch in batches)
    assert total_samples == 10


@pytest.fixture
def mock_args():
    """Create mock arguments for multitask trainer."""
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
def mock_multitask_dataset():
    """Create mock multitask dataset."""
    dataset = []
    for i in range(8):
        dataset.append(
            {
                "bboxes": torch.randn(3, 4),
                "labels": torch.randint(0, 5, (3,)),
                "mask": torch.ones(3, dtype=torch.bool),
            }
        )

    # Add num_tasks attribute
    class MockDataset(list):
        def __init__(self, data):
            super().__init__(data)
            self.num_tasks = 2

    return MockDataset(dataset)


@pytest.fixture
def mock_optimizer(mock_model):
    """Create mock optimizer."""
    return torch.optim.Adam(mock_model.parameters(), lr=0.001)


@patch("layoutformer_pp.trainer.multitask_trainer.wandb")
@patch("layoutformer_pp.trainer.basic_trainer.utils.init_experiment")
@patch("layoutformer_pp.trainer.basic_trainer.utils.log_hyperparameters")
def test_multitask_trainer_initialization(
    mock_log_hyperparam,
    mock_init_exp,
    mock_wandb,
    mock_args,
    mock_tokenizer,
    mock_model,
    mock_seq_processor,
    mock_multitask_dataset,
    mock_optimizer,
):
    """Test MultiTaskTrainer initialization."""
    mock_log_hyperparam.return_value = {"learning_rate": 0.001}

    trainer = MultiTaskTrainer(
        task_name="test_task",
        args=mock_args,
        tokenizer=mock_tokenizer,
        model=mock_model,
        seq_processor=mock_seq_processor,
        train_dataset=mock_multitask_dataset,
        val_dataset=mock_multitask_dataset,
        optimizer=mock_optimizer,
        single_task_per_batch=False,
        save_vocab=False,
    )

    assert trainer._single_task_per_batch is False
    assert hasattr(trainer, "train_dataloader")


@patch("layoutformer_pp.trainer.multitask_trainer.wandb")
@patch("layoutformer_pp.trainer.basic_trainer.utils.init_experiment")
@patch("layoutformer_pp.trainer.basic_trainer.utils.log_hyperparameters")
def test_multitask_trainer_single_task_per_batch(
    mock_log_hyperparam,
    mock_init_exp,
    mock_wandb,
    mock_args,
    mock_tokenizer,
    mock_model,
    mock_seq_processor,
    mock_multitask_dataset,
    mock_optimizer,
):
    """Test MultiTaskTrainer with single task per batch."""
    mock_log_hyperparam.return_value = {"learning_rate": 0.001}

    trainer = MultiTaskTrainer(
        task_name="test_task",
        args=mock_args,
        tokenizer=mock_tokenizer,
        model=mock_model,
        seq_processor=mock_seq_processor,
        train_dataset=mock_multitask_dataset,
        val_dataset=mock_multitask_dataset,
        optimizer=mock_optimizer,
        single_task_per_batch=True,
        single_task_per_batch_task_sample_weights="1.0,2.0",
        save_vocab=False,
    )

    assert trainer._single_task_per_batch is True
    assert trainer._single_task_per_batch_task_sample_weights == "1.0,2.0"


@patch("layoutformer_pp.trainer.multitask_trainer.wandb")
@patch("layoutformer_pp.trainer.basic_trainer.utils.init_experiment")
@patch("layoutformer_pp.trainer.basic_trainer.utils.log_hyperparameters")
def test_multitask_trainer_vocab_saving(
    mock_log_hyperparam,
    mock_init_exp,
    mock_wandb,
    mock_args,
    mock_tokenizer,
    mock_model,
    mock_seq_processor,
    mock_multitask_dataset,
    mock_optimizer,
):
    """Test vocabulary saving in MultiTaskTrainer."""
    mock_log_hyperparam.return_value = {"learning_rate": 0.001}

    _ = MultiTaskTrainer(
        task_name="test_task",
        args=mock_args,
        tokenizer=mock_tokenizer,
        model=mock_model,
        seq_processor=mock_seq_processor,
        train_dataset=mock_multitask_dataset,
        val_dataset=mock_multitask_dataset,
        optimizer=mock_optimizer,
        save_vocab=True,
    )

    vocab_path = os.path.join(mock_args.out_dir, "vocab.json")
    assert os.path.exists(vocab_path)


@patch("layoutformer_pp.trainer.multitask_trainer.wandb")
@patch("layoutformer_pp.trainer.basic_trainer.utils.init_experiment")
@patch("layoutformer_pp.trainer.basic_trainer.utils.log_hyperparameters")
def test_multitask_trainer_update_best_miou(
    mock_log_hyperparam,
    mock_init_exp,
    mock_wandb,
    mock_args,
    mock_tokenizer,
    mock_model,
    mock_seq_processor,
    mock_multitask_dataset,
    mock_optimizer,
):
    """Test best mIoU update logic."""
    mock_log_hyperparam.return_value = {"learning_rate": 0.001}

    trainer = MultiTaskTrainer(
        task_name="test_task",
        args=mock_args,
        tokenizer=mock_tokenizer,
        model=mock_model,
        seq_processor=mock_seq_processor,
        train_dataset=mock_multitask_dataset,
        val_dataset=mock_multitask_dataset,
        optimizer=mock_optimizer,
        save_vocab=False,
    )

    # Initialize best mIoU tracking
    best_miou = {
        "task1": {"value": 0.5, "epoch": 0},
        "task2": {"value": 0.3, "epoch": 0},
    }

    # Update with better values
    curr_miou = {"task1": 0.6, "task2": 0.4}
    trainer.update_best_miou(1, curr_miou, best_miou)

    assert best_miou["task1"]["value"] == 0.6
    assert best_miou["task1"]["epoch"] == 1
    assert best_miou["task2"]["value"] == 0.4
    assert best_miou["task2"]["epoch"] == 1

    # Update with worse values (should not change)
    curr_miou = {"task1": 0.4, "task2": 0.2}
    trainer.update_best_miou(2, curr_miou, best_miou)

    assert best_miou["task1"]["value"] == 0.6
    assert best_miou["task1"]["epoch"] == 1


@patch("layoutformer_pp.trainer.multitask_trainer.wandb")
@patch("layoutformer_pp.trainer.basic_trainer.utils.init_experiment")
@patch("layoutformer_pp.trainer.basic_trainer.utils.log_hyperparameters")
def test_multitask_trainer_checkpointing(
    mock_log_hyperparam,
    mock_init_exp,
    mock_wandb,
    mock_args,
    mock_tokenizer,
    mock_model,
    mock_seq_processor,
    mock_multitask_dataset,
    mock_optimizer,
):
    """Test multitask trainer checkpointing."""
    mock_log_hyperparam.return_value = {"learning_rate": 0.001}

    trainer = MultiTaskTrainer(
        task_name="test_task",
        args=mock_args,
        tokenizer=mock_tokenizer,
        model=mock_model,
        seq_processor=mock_seq_processor,
        train_dataset=mock_multitask_dataset,
        val_dataset=mock_multitask_dataset,
        optimizer=mock_optimizer,
        save_vocab=False,
    )

    best_miou = {
        "task1": {"value": 0.5, "epoch": 0},
        "task2": {"value": 0.3, "epoch": 0},
    }

    trainer.do_checkpointing(epoch=0, is_best=True, task_best_miou=best_miou)

    # Check that files are created
    assert os.path.exists(os.path.join(mock_args.out_dir, "best_epoch"))
    assert os.path.exists(os.path.join(mock_args.out_dir, "best_miou.json"))

    # Read and verify content
    with open(os.path.join(mock_args.out_dir, "best_epoch")) as f:
        assert f.read() == "0"

    import json

    with open(os.path.join(mock_args.out_dir, "best_miou.json")) as f:
        saved_miou = json.load(f)
        assert saved_miou == best_miou


@patch("layoutformer_pp.trainer.multitask_trainer.wandb")
@patch("layoutformer_pp.trainer.basic_trainer.utils.init_experiment")
@patch("layoutformer_pp.trainer.basic_trainer.utils.log_hyperparameters")
def test_multitask_trainer_dataloader_batch_sampler(
    mock_log_hyperparam,
    mock_init_exp,
    mock_wandb,
    mock_args,
    mock_tokenizer,
    mock_model,
    mock_seq_processor,
    mock_multitask_dataset,
    mock_optimizer,
):
    """Test that dataloader uses MultiTaskBatchSampler when enabled."""
    mock_log_hyperparam.return_value = {"learning_rate": 0.001}

    trainer = MultiTaskTrainer(
        task_name="test_task",
        args=mock_args,
        tokenizer=mock_tokenizer,
        model=mock_model,
        seq_processor=mock_seq_processor,
        train_dataset=mock_multitask_dataset,
        val_dataset=mock_multitask_dataset,
        optimizer=mock_optimizer,
        single_task_per_batch=True,
        single_task_per_batch_task_sample_weights=None,
        save_vocab=False,
    )

    # Dataloader should be set up
    assert trainer.train_dataloader is not None
    assert trainer.val_dataloader is not None


@patch("layoutformer_pp.trainer.multitask_trainer.wandb")
@patch("layoutformer_pp.trainer.basic_trainer.utils.init_experiment")
@patch("layoutformer_pp.trainer.basic_trainer.utils.log_hyperparameters")
def test_multitask_trainer_call_runs_one_epoch(
    mock_log_hyperparam,
    mock_init_exp,
    mock_wandb,
):
    """Run a minimal train/eval loop to cover __call__."""
    mock_log_hyperparam.return_value = {"learning_rate": 0.001}

    with tempfile.TemporaryDirectory() as tmpdir:
        args = SimpleNamespace(
            out_dir=tmpdir,
            batch_size=1,
            eval_batch_size=1,
            epoch=1,
            train_log_step=1,
            gradient_accumulation=1,
            enable_clip_gradient=False,
            clip_gradient=1.0,
            load_train_ckpt=False,
            train_ckpt_path=None,
            max_num_elements=1,
            discrete_x_grid=16,
            discrete_y_grid=16,
            bbox_format="ltrb",
        )

        tokenizer = LayoutTransformerTokenizer(tokens=["label_1", "0", "1"])
        model = nn.Linear(1, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        class DummyDataset(torch.utils.data.Dataset):
            def __init__(self, tasks, num_items):
                self.tasks = tasks
                self.num_tasks = len(tasks)
                self._curr_task = tasks[0]
                self.seq_processor = SimpleNamespace(error_label_id=0)
                self.data = list(range(num_items))

            def switch_task(self, task):
                self._curr_task = task

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                return {
                    "in_str": "label_1 0 0 1 1",
                    "out_str": "label_1 0 0 1 1",
                    "gold_labels": torch.tensor([1]),
                    "gold_bboxes": torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
                    "task_name": self._curr_task,
                    "task_id": 0,
                    "name": f"sample-{idx}",
                }

        tasks = ["task_a", "task_b"]
        train_dataset = DummyDataset(tasks, num_items=2)
        val_dataset = DummyDataset(tasks, num_items=1)

        trainer = MultiTaskTrainer(
            task_name="test_task",
            args=args,
            tokenizer=tokenizer,
            model=model,
            seq_processor=SimpleNamespace(error_label_id=0),
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            optimizer=optimizer,
            checkpoint_measure="eval_loss",
            save_vocab=False,
        )

        def train_step(step_model, data, step_tokenizer, device):
            return step_model.weight.sum()

        def eval_step(step_model, data, seq_processor, step_tokenizer, device):
            batch_size = len(data["task_name"])
            pred_bboxes = torch.zeros(batch_size, 1, 4)
            pred_labels = torch.ones(batch_size, 1, dtype=torch.long)
            pred_mask = torch.ones(batch_size, 1, dtype=torch.bool)
            gold_bboxes = torch.zeros(batch_size, 1, 4)
            gold_labels = torch.ones(batch_size, 1, dtype=torch.long)
            gold_mask = torch.ones(batch_size, 1, dtype=torch.bool)
            return torch.tensor(1.0), {
                "pred_bboxes": pred_bboxes,
                "pred_labels": pred_labels,
                "pred_mask": pred_mask,
                "gold_bboxes": gold_bboxes,
                "gold_labels": gold_labels,
                "gold_mask": gold_mask,
            }

        with patch(
            "layoutformer_pp.trainer.utils.metrics.compute_maximum_iou",
            return_value=1.0,
        ):
            trainer(train_step, eval_step, tasks=tasks, eval_interval=1)

        assert os.path.exists(os.path.join(args.out_dir, "best_epoch"))


@patch("layoutformer_pp.trainer.multitask_trainer.wandb")
@patch("layoutformer_pp.trainer.basic_trainer.utils.init_experiment")
@patch("layoutformer_pp.trainer.basic_trainer.utils.log_hyperparameters")
def test_multitask_trainer_call_handles_none_predictions(
    mock_log_hyperparam,
    mock_init_exp,
    mock_wandb,
):
    """Cover eval_step_pred None path plus clip/scheduler branches."""
    mock_log_hyperparam.return_value = {"learning_rate": 0.001}

    with tempfile.TemporaryDirectory() as tmpdir:
        args = SimpleNamespace(
            out_dir=tmpdir,
            batch_size=1,
            eval_batch_size=1,
            epoch=1,
            train_log_step=1,
            gradient_accumulation=1,
            enable_clip_gradient=True,
            clip_gradient=1.0,
            load_train_ckpt=False,
            train_ckpt_path=None,
            max_num_elements=1,
            discrete_x_grid=16,
            discrete_y_grid=16,
            bbox_format="ltrb",
        )

        tokenizer = LayoutTransformerTokenizer(tokens=["label_1", "0", "1"])
        model = nn.Linear(1, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

        class DummyDataset(torch.utils.data.Dataset):
            def __init__(self, tasks, num_items):
                self.tasks = tasks
                self.num_tasks = len(tasks)
                self._curr_task = tasks[0]
                self.seq_processor = SimpleNamespace(error_label_id=0)
                self.data = list(range(num_items))

            def switch_task(self, task):
                self._curr_task = task

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                return {
                    "in_str": "label_1 0 0 1 1",
                    "out_str": "label_1 0 0 1 1",
                    "gold_labels": torch.tensor([1]),
                    "gold_bboxes": torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
                    "task_name": self._curr_task,
                    "task_id": 0,
                    "name": f"sample-{idx}",
                }

        tasks = ["task_a", "task_b"]
        train_dataset = DummyDataset(tasks, num_items=2)
        val_dataset = DummyDataset(tasks, num_items=1)

        trainer = MultiTaskTrainer(
            task_name="test_task",
            args=args,
            tokenizer=tokenizer,
            model=model,
            seq_processor=SimpleNamespace(error_label_id=0),
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            optimizer=optimizer,
            scheduler=scheduler,
            checkpoint_measure="eval_loss",
            save_vocab=False,
        )

        def train_step(step_model, data, step_tokenizer, device):
            return step_model.weight.sum()

        def eval_step(step_model, data, seq_processor, step_tokenizer, device):
            return torch.tensor(1.0), None

        trainer(train_step, eval_step, tasks=tasks, eval_interval=1)

        assert os.path.exists(os.path.join(args.out_dir, "epoch_0_checkpoint.pth.tar"))


@patch.dict(os.environ, {"LOCAL_RANK": "0", "WORLD_SIZE": "1"}, clear=False)
@patch("layoutformer_pp.trainer.multitask_trainer.wandb")
@patch("layoutformer_pp.trainer.ds_trainer.wandb")
@patch("layoutformer_pp.trainer.basic_trainer.wandb")
@patch("layoutformer_pp.trainer.basic_trainer.utils.init_experiment")
@patch("layoutformer_pp.trainer.basic_trainer.utils.log_hyperparameters")
@patch("layoutformer_pp.trainer.ds_trainer.utils.load_arguments")
@patch("layoutformer_pp.trainer.ds_trainer.deepspeed")
@patch("torch.distributed.barrier")
def test_ds_multitask_trainer_call_runs_one_epoch(
    mock_barrier,
    mock_deepspeed,
    mock_load_args,
    mock_log_hyperparam,
    mock_init_exp,
    mock_basic_wandb,
    mock_ds_wandb,
    mock_wandb,
):
    """Run DSMultiTaskTrainer for one epoch to cover deepspeed paths."""
    ds_config = {
        "train_micro_batch_size_per_gpu": 1,
        "gradient_accumulation_steps": 1,
        "scheduler": {"params": {"warmup_num_steps": 10}},
    }
    mock_load_args.return_value = ds_config
    mock_log_hyperparam.return_value = {"learning_rate": 0.001}

    mock_engine = MagicMock()
    mock_engine.local_rank = 0
    mock_engine.backward = Mock()
    mock_engine.step = Mock()
    mock_engine.train = Mock()
    mock_engine.eval = Mock()
    mock_engine.save_checkpoint = Mock()
    mock_engine.load_checkpoint = Mock(return_value=(None, {}))
    mock_deepspeed.initialize = Mock(return_value=(mock_engine, Mock(), None, None))
    mock_deepspeed.init_distributed = Mock()

    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = os.path.join(tmpdir, "ds_config.json")
        with open(config_path, "w") as f:
            import json

            json.dump(ds_config, f)

        args = SimpleNamespace(
            out_dir=tmpdir,
            batch_size=1,
            eval_batch_size=1,
            epoch=1,
            train_log_step=1,
            gradient_accumulation=1,
            enable_clip_gradient=False,
            clip_gradient=1.0,
            load_train_ckpt=False,
            train_ckpt_path=None,
            ds_ckpt_tag="unit",
            max_num_elements=1,
            discrete_x_grid=16,
            discrete_y_grid=16,
            bbox_format="ltrb",
            backend="nccl",
            deepscale_config=config_path,
        )

        tokenizer = LayoutTransformerTokenizer(tokens=["label_1", "0", "1"])
        model = nn.Linear(1, 1)
        optimizer = Mock()

        class DummyDataset(torch.utils.data.Dataset):
            def __init__(self, tasks, num_items):
                self.tasks = tasks
                self.num_tasks = len(tasks)
                self._curr_task = tasks[0]
                self.seq_processor = SimpleNamespace(error_label_id=0)
                self.data = list(range(num_items))

            def switch_task(self, task):
                self._curr_task = task

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                return {
                    "in_str": "label_1 0 0 1 1",
                    "out_str": "label_1 0 0 1 1",
                    "gold_labels": torch.tensor([1]),
                    "gold_bboxes": torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
                    "task_name": self._curr_task,
                    "task_id": 0,
                    "name": f"sample-{idx}",
                }

        tasks = ["task_a"]
        train_dataset = DummyDataset(tasks, num_items=2)
        val_dataset = DummyDataset(tasks, num_items=1)

        trainer = DSMultiTaskTrainer(
            task_name="test_task",
            args=args,
            tokenizer=tokenizer,
            model=model,
            seq_processor=SimpleNamespace(error_label_id=0),
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            optimizer=optimizer,
            checkpoint_measure="eval_loss",
            d2c_fn=lambda x: x,
            task_config={},
            save_vocab=False,
        )

        trainer.ckpt_measurement.update = Mock(return_value=True)

        def train_step(step_model, data, step_tokenizer, device):
            return torch.tensor(1.0)

        def eval_step(step_model, data, seq_processor, step_tokenizer, device):
            batch_size = len(data["task_name"])
            pred_bboxes = torch.zeros(batch_size, 1, 4)
            pred_labels = torch.ones(batch_size, 1, dtype=torch.long)
            pred_mask = torch.ones(batch_size, 1, dtype=torch.bool)
            gold_bboxes = torch.zeros(batch_size, 1, 4)
            gold_labels = torch.ones(batch_size, 1, dtype=torch.long)
            gold_mask = torch.ones(batch_size, 1, dtype=torch.bool)
            return torch.tensor(1.0), {
                "pred_bboxes": pred_bboxes,
                "pred_labels": pred_labels,
                "pred_mask": pred_mask,
                "gold_bboxes": gold_bboxes,
                "gold_labels": gold_labels,
                "gold_mask": gold_mask,
            }

        trainer(train_step, eval_step, tasks=tasks, eval_interval=1)

        mock_engine.save_checkpoint.assert_called()
        assert os.path.exists(os.path.join(args.out_dir, "best_epoch"))
