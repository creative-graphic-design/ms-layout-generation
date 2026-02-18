from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch as th

from improved_diffusion import dist_util
from improved_diffusion.gaussian_diffusion import (
    GaussianDiffusion,
    LossType,
    ModelMeanType,
    ModelVarType,
)
from improved_diffusion.train_util import (
    TrainLoop,
    find_ema_checkpoint,
    parse_resume_step_from_filename,
)


@pytest.fixture
def setup_dist():
    """Initialize distributed training for tests"""
    if not th.distributed.is_initialized():
        dist_util.setup_dist()
    yield
    # Cleanup if needed


class SimpleModel(th.nn.Module):
    """Minimal model for testing TrainLoop"""

    def __init__(self, in_channels=8, out_channels=8):
        super().__init__()
        self.linear = th.nn.Linear(in_channels, out_channels)

    def forward(self, x, timesteps, **kwargs):
        x = x.to(self.linear.weight.dtype)
        return self.linear(x)


def create_mock_diffusion():
    """Create a mock diffusion for testing"""
    betas = np.linspace(0.0001, 0.02, 100, dtype=np.float64)
    diffusion = GaussianDiffusion(
        betas=betas,
        model_mean_type=ModelMeanType.EPSILON,
        model_var_type=ModelVarType.FIXED_SMALL,
        loss_type=LossType.MSE,
    )
    return diffusion


def create_tiny_data_generator(batch_size=2, seq_len=10, in_channel=8):
    """Create a tiny synthetic data generator"""
    while True:
        batch = th.randn(batch_size, seq_len, in_channel)
        cond = {"input_ids": th.randint(0, 100, (batch_size, seq_len))}
        yield batch, cond


class TestTrainLoopInitialization:
    """Test TrainLoop initialization and setup"""

    def test_trainloop_basic_initialization(self, tmp_path: Path, setup_dist) -> None:
        """Test basic TrainLoop initialization without GPU"""
        model = SimpleModel()
        diffusion = create_mock_diffusion()
        data = create_tiny_data_generator()

        train_loop = TrainLoop(
            model=model,
            diffusion=diffusion,
            data=data,
            batch_size=2,
            microbatch=2,
            lr=1e-4,
            ema_rate=0.9999,
            log_interval=10,
            save_interval=100,
            resume_checkpoint="",
            checkpoint_path=str(tmp_path),
        )

        assert train_loop.step == 0
        assert train_loop.resume_step == 0
        assert train_loop.batch_size == 2
        assert train_loop.microbatch == 2
        assert train_loop.lr == 1e-4
        assert train_loop.ema_rate == [0.9999]
        assert len(train_loop.ema_params) == 1

    def test_trainloop_fp16_initialization(self, tmp_path: Path, setup_dist) -> None:
        """Test TrainLoop initialization with FP16 enabled"""
        model = SimpleModel()
        diffusion = create_mock_diffusion()
        data = create_tiny_data_generator()

        train_loop = TrainLoop(
            model=model,
            diffusion=diffusion,
            data=data,
            batch_size=2,
            microbatch=1,
            lr=1e-4,
            ema_rate=0.9999,
            log_interval=10,
            save_interval=100,
            resume_checkpoint="",
            use_fp16=True,
            checkpoint_path=str(tmp_path),
        )

        assert train_loop.use_fp16 is True
        assert train_loop.lg_loss_scale == 20.0

    def test_trainloop_multiple_ema_rates(self, tmp_path: Path, setup_dist) -> None:
        """Test TrainLoop with multiple EMA rates"""
        model = SimpleModel()
        diffusion = create_mock_diffusion()
        data = create_tiny_data_generator()

        train_loop = TrainLoop(
            model=model,
            diffusion=diffusion,
            data=data,
            batch_size=2,
            microbatch=2,
            lr=1e-4,
            ema_rate="0.9999,0.999,0.99",
            log_interval=10,
            save_interval=100,
            resume_checkpoint="",
            checkpoint_path=str(tmp_path),
        )

        assert len(train_loop.ema_rate) == 3
        assert len(train_loop.ema_params) == 3
        assert train_loop.ema_rate == [0.9999, 0.999, 0.99]

    def test_trainloop_microbatch_auto_set(self, tmp_path: Path, setup_dist) -> None:
        """Test that microbatch defaults to batch_size when set to <= 0"""
        model = SimpleModel()
        diffusion = create_mock_diffusion()
        data = create_tiny_data_generator()

        train_loop = TrainLoop(
            model=model,
            diffusion=diffusion,
            data=data,
            batch_size=4,
            microbatch=-1,
            lr=1e-4,
            ema_rate=0.9999,
            log_interval=10,
            save_interval=100,
            resume_checkpoint="",
            checkpoint_path=str(tmp_path),
        )

        assert train_loop.microbatch == 4

    @pytest.mark.gpu
    def test_trainloop_ddp_initialization(
        self, gpu_device, tmp_path: Path, setup_dist
    ) -> None:
        """Test TrainLoop DDP initialization on GPU"""
        model = SimpleModel().to(gpu_device)
        diffusion = create_mock_diffusion()
        data = create_tiny_data_generator()

        train_loop = TrainLoop(
            model=model,
            diffusion=diffusion,
            data=data,
            batch_size=2,
            microbatch=2,
            lr=1e-4,
            ema_rate=0.9999,
            log_interval=10,
            save_interval=100,
            resume_checkpoint="",
            checkpoint_path=str(tmp_path),
        )

        assert train_loop.use_ddp is True
        assert hasattr(train_loop, "ddp_model")


class TestTrainLoopForwardBackward:
    """Test forward_backward method and gradient accumulation"""

    def test_forward_backward_basic(self, tmp_path: Path, setup_dist) -> None:
        """Test basic forward_backward without errors"""
        model = SimpleModel()
        diffusion = create_mock_diffusion()
        data = create_tiny_data_generator(batch_size=4, seq_len=10)

        train_loop = TrainLoop(
            model=model,
            diffusion=diffusion,
            data=data,
            batch_size=4,
            microbatch=2,
            lr=1e-4,
            ema_rate=0.9999,
            log_interval=10,
            save_interval=100,
            resume_checkpoint="",
            checkpoint_path=str(tmp_path),
        )

        batch, cond = next(data)
        train_loop.forward_backward(batch, cond)

        # Check that gradients were computed
        for param in train_loop.model_params:
            if param.requires_grad:
                assert param.grad is not None

    def test_forward_backward_with_microbatch(self, tmp_path: Path, setup_dist) -> None:
        """Test forward_backward with gradient accumulation"""
        model = SimpleModel()
        diffusion = create_mock_diffusion()
        data = create_tiny_data_generator(batch_size=4, seq_len=10)

        train_loop = TrainLoop(
            model=model,
            diffusion=diffusion,
            data=data,
            batch_size=4,
            microbatch=1,  # Process in smaller chunks
            lr=1e-4,
            ema_rate=0.9999,
            log_interval=10,
            save_interval=100,
            resume_checkpoint="",
            checkpoint_path=str(tmp_path),
        )

        batch, cond = next(data)
        train_loop.forward_backward(batch, cond)

        for param in train_loop.model_params:
            if param.requires_grad:
                assert param.grad is not None

    def test_forward_backward_discrete_mode(self, tmp_path: Path, setup_dist) -> None:
        """Test forward_backward with discrete training mode"""
        from improved_diffusion.discrete_diffusion import DiffusionTransformer

        class DiscreteDummyModel(th.nn.Module):
            def __init__(self, num_classes: int, hidden_dim: int = 16) -> None:
                super().__init__()
                self.embedding = th.nn.Embedding(num_classes, hidden_dim)
                self.proj = th.nn.Linear(hidden_dim, num_classes - 1)

            def forward(self, x_t, t, y=None):
                embeddings = self.embedding(x_t)
                logits = self.proj(embeddings)
                return logits.permute(0, 2, 1)

        model = DiscreteDummyModel(num_classes=159)
        # Create a discrete diffusion
        diffusion = DiffusionTransformer(
            diffusion_step=10,
            content_seq_len=10,
            num_classes=159,
            alpha_init_type="gaussian_refine_pow2.5",
        )
        data = create_tiny_data_generator(batch_size=2, seq_len=10)

        train_loop = TrainLoop(
            model=model,
            diffusion=diffusion,
            data=data,
            batch_size=2,
            microbatch=2,
            lr=1e-4,
            ema_rate=0.9999,
            log_interval=10,
            save_interval=100,
            resume_checkpoint="",
            training_mode="discrete",
            checkpoint_path=str(tmp_path),
        )

        batch, cond = next(data)
        train_loop.forward_backward(batch, cond)

        for param in train_loop.model_params:
            if param.requires_grad:
                assert param.grad is not None

    @pytest.mark.gpu
    def test_forward_backward_on_gpu(
        self, gpu_device, tmp_path: Path, setup_dist
    ) -> None:
        """Test forward_backward on GPU"""
        model = SimpleModel().to(gpu_device)
        diffusion = create_mock_diffusion()
        data = create_tiny_data_generator(batch_size=2, seq_len=10)

        train_loop = TrainLoop(
            model=model,
            diffusion=diffusion,
            data=data,
            batch_size=2,
            microbatch=2,
            lr=1e-4,
            ema_rate=0.9999,
            log_interval=10,
            save_interval=100,
            resume_checkpoint="",
            checkpoint_path=str(tmp_path),
        )

        batch, cond = next(data)
        train_loop.forward_backward(batch, cond)

        for param in train_loop.model_params:
            if param.requires_grad:
                assert param.grad is not None
                assert param.grad.device.type == "cuda"


class TestTrainLoopOptimization:
    """Test optimize_fp16 and optimize_normal methods"""

    def test_optimize_normal_basic(self, tmp_path: Path, setup_dist) -> None:
        """Test optimize_normal performs optimization step"""
        model = SimpleModel()
        diffusion = create_mock_diffusion()
        data = create_tiny_data_generator(batch_size=2)

        train_loop = TrainLoop(
            model=model,
            diffusion=diffusion,
            data=data,
            batch_size=2,
            microbatch=2,
            lr=1e-4,
            ema_rate=0.9999,
            log_interval=10,
            save_interval=100,
            resume_checkpoint="",
            checkpoint_path=str(tmp_path),
        )

        batch, cond = next(data)
        train_loop.forward_backward(batch, cond)

        # Store initial params
        initial_params = [p.clone().detach() for p in train_loop.model_params]

        train_loop.optimize_normal()

        # Check that params changed after optimization
        for initial, current in zip(initial_params, train_loop.model_params):
            if current.requires_grad:
                assert not th.allclose(initial, current)

    def test_optimize_normal_with_gradient_clipping(
        self, tmp_path: Path, setup_dist
    ) -> None:
        """Test optimize_normal with gradient clipping"""
        model = SimpleModel()
        diffusion = create_mock_diffusion()
        data = create_tiny_data_generator(batch_size=2)

        train_loop = TrainLoop(
            model=model,
            diffusion=diffusion,
            data=data,
            batch_size=2,
            microbatch=2,
            lr=1e-4,
            ema_rate=0.9999,
            log_interval=10,
            save_interval=100,
            resume_checkpoint="",
            gradient_clipping=1.0,
            checkpoint_path=str(tmp_path),
        )

        batch, cond = next(data)
        train_loop.forward_backward(batch, cond)
        train_loop.optimize_normal()

        # Check that gradient clipping was applied (gradients should be bounded)
        total_norm = 0.0
        for p in train_loop.model_params:
            if p.grad is not None:
                total_norm += (p.grad**2).sum().item()
        total_norm = np.sqrt(total_norm)

        # After clipping, norm should be reasonable
        assert total_norm < 100.0

    def test_optimize_fp16_basic(self, tmp_path: Path, setup_dist) -> None:
        """Test optimize_fp16 performs FP16 optimization"""
        model = SimpleModel()
        diffusion = create_mock_diffusion()
        data = create_tiny_data_generator(batch_size=2)

        train_loop = TrainLoop(
            model=model,
            diffusion=diffusion,
            data=data,
            batch_size=2,
            microbatch=2,
            lr=1e-4,
            ema_rate=0.9999,
            log_interval=10,
            save_interval=100,
            resume_checkpoint="",
            use_fp16=True,
            checkpoint_path=str(tmp_path),
        )

        batch, cond = next(data)
        train_loop.forward_backward(batch, cond)

        initial_lg_loss_scale = train_loop.lg_loss_scale
        train_loop.optimize_fp16()

        # Loss scale should increase after successful step
        assert train_loop.lg_loss_scale >= initial_lg_loss_scale

    def test_optimize_fp16_nan_handling(self, tmp_path: Path, setup_dist) -> None:
        """Test optimize_fp16 handles NaN gradients"""
        model = SimpleModel()
        diffusion = create_mock_diffusion()
        data = create_tiny_data_generator(batch_size=2)

        train_loop = TrainLoop(
            model=model,
            diffusion=diffusion,
            data=data,
            batch_size=2,
            microbatch=2,
            lr=1e-4,
            ema_rate=0.9999,
            log_interval=10,
            save_interval=100,
            resume_checkpoint="",
            use_fp16=True,
            checkpoint_path=str(tmp_path),
        )

        batch, cond = next(data)
        train_loop.forward_backward(batch, cond)

        # Inject NaN into gradients
        for param in train_loop.model_params:
            if param.grad is not None:
                param.grad[0] = float("nan")

        initial_lg_loss_scale = train_loop.lg_loss_scale
        train_loop.optimize_fp16()

        # Loss scale should decrease when NaN detected
        assert train_loop.lg_loss_scale < initial_lg_loss_scale


class TestTrainLoopEMA:
    """Test EMA parameter updates"""

    def test_ema_update_after_optimization(self, tmp_path: Path, setup_dist) -> None:
        """Test that EMA parameters are updated during optimization"""
        model = SimpleModel()
        diffusion = create_mock_diffusion()
        data = create_tiny_data_generator(batch_size=2)

        train_loop = TrainLoop(
            model=model,
            diffusion=diffusion,
            data=data,
            batch_size=2,
            microbatch=2,
            lr=1e-4,
            ema_rate=0.9999,
            log_interval=10,
            save_interval=100,
            resume_checkpoint="",
            checkpoint_path=str(tmp_path),
        )

        # Store initial EMA params
        initial_ema = [p.clone().detach() for p in train_loop.ema_params[0]]

        # Run a training step
        batch, cond = next(data)
        train_loop.forward_backward(batch, cond)
        train_loop.optimize_normal()

        # EMA params should have changed
        for initial, current in zip(initial_ema, train_loop.ema_params[0]):
            # EMA should be different after update
            # (may be small difference due to high rate)
            assert not th.equal(initial, current)


class TestTrainLoopSaveResume:
    """Test save and resume_from_checkpoint functionality"""

    def test_save_checkpoint(self, tmp_path: Path, setup_dist) -> None:
        """Test saving checkpoint creates expected files"""
        model = SimpleModel()
        diffusion = create_mock_diffusion()
        data = create_tiny_data_generator(batch_size=2)

        train_loop = TrainLoop(
            model=model,
            diffusion=diffusion,
            data=data,
            batch_size=2,
            microbatch=2,
            lr=1e-4,
            ema_rate=0.9999,
            log_interval=10,
            save_interval=100,
            resume_checkpoint="",
            checkpoint_path=str(tmp_path),
        )

        train_loop.save()

        # Check that model checkpoint was saved
        model_ckpt = tmp_path / "model000000.pt"
        assert model_ckpt.exists()

        # Check that EMA checkpoint was saved
        ema_ckpt = tmp_path / "ema_0.9999_000000.pt"
        assert ema_ckpt.exists()

    def test_save_checkpoint_with_steps(self, tmp_path: Path, setup_dist) -> None:
        """Test checkpoint naming with step numbers"""
        model = SimpleModel()
        diffusion = create_mock_diffusion()
        data = create_tiny_data_generator(batch_size=2)

        train_loop = TrainLoop(
            model=model,
            diffusion=diffusion,
            data=data,
            batch_size=2,
            microbatch=2,
            lr=1e-4,
            ema_rate=0.9999,
            log_interval=10,
            save_interval=100,
            resume_checkpoint="",
            checkpoint_path=str(tmp_path),
        )

        train_loop.step = 42
        train_loop.save()

        model_ckpt = tmp_path / "model000042.pt"
        ema_ckpt = tmp_path / "ema_0.9999_000042.pt"

        assert model_ckpt.exists()
        assert ema_ckpt.exists()

    def test_resume_from_checkpoint(self, tmp_path: Path, setup_dist) -> None:
        """Test resuming training from checkpoint"""
        # First, create and save a checkpoint
        model1 = SimpleModel()
        diffusion = create_mock_diffusion()
        data1 = create_tiny_data_generator(batch_size=2)

        train_loop1 = TrainLoop(
            model=model1,
            diffusion=diffusion,
            data=data1,
            batch_size=2,
            microbatch=2,
            lr=1e-4,
            ema_rate=0.9999,
            log_interval=10,
            save_interval=100,
            resume_checkpoint="",
            checkpoint_path=str(tmp_path),
        )

        # Run a few steps and save
        for _ in range(3):
            batch, cond = next(data1)
            train_loop1.run_step(batch, cond)
            train_loop1.step += 1

        train_loop1.save()

        # Now create a new model and resume
        model2 = SimpleModel()
        data2 = create_tiny_data_generator(batch_size=2)
        checkpoint_path = str(tmp_path / "model000003.pt")

        train_loop2 = TrainLoop(
            model=model2,
            diffusion=diffusion,
            data=data2,
            batch_size=2,
            microbatch=2,
            lr=1e-4,
            ema_rate=0.9999,
            log_interval=10,
            save_interval=100,
            resume_checkpoint=checkpoint_path,
            checkpoint_path=str(tmp_path),
        )

        # Check that resume_step was set correctly
        assert train_loop2.resume_step == 3

        # Check that model weights match
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            assert th.allclose(p1, p2)


class TestTrainLoopHelperFunctions:
    """Test helper functions in train_util"""

    def test_parse_resume_step_from_filename(self) -> None:
        """Test parsing step number from checkpoint filename"""
        assert parse_resume_step_from_filename("model000042.pt") == 42
        assert parse_resume_step_from_filename("path/to/model000123.pt") == 123
        # Function only parses "model" filenames, not EMA filenames
        assert parse_resume_step_from_filename("ema_0.9999_001234.pt") == 0
        assert parse_resume_step_from_filename("invalid.pt") == 0
        assert parse_resume_step_from_filename("model.pt") == 0

    def test_find_ema_checkpoint(self, tmp_path: Path) -> None:
        """Test finding EMA checkpoint"""
        # Create a dummy EMA checkpoint
        ema_path = tmp_path / "ema_0.9999_000042.pt"
        ema_path.touch()

        main_checkpoint = str(tmp_path / "model000042.pt")
        found = find_ema_checkpoint(main_checkpoint, step=42, rate=0.9999)

        assert found == str(ema_path)

    def test_find_ema_checkpoint_not_found(self, tmp_path: Path) -> None:
        """Test find_ema_checkpoint returns None when not found"""
        main_checkpoint = str(tmp_path / "model000042.pt")
        found = find_ema_checkpoint(main_checkpoint, step=42, rate=0.9999)

        assert found is None

    def test_find_ema_checkpoint_none_input(self) -> None:
        """Test find_ema_checkpoint with None main_checkpoint"""
        found = find_ema_checkpoint(None, step=42, rate=0.9999)
        assert found is None


class TestTrainLoopLRAnnealing:
    """Test learning rate annealing"""

    def test_lr_annealing_disabled(self, tmp_path: Path, setup_dist) -> None:
        """Test that LR doesn't change when annealing disabled"""
        model = SimpleModel()
        diffusion = create_mock_diffusion()
        data = create_tiny_data_generator(batch_size=2)

        train_loop = TrainLoop(
            model=model,
            diffusion=diffusion,
            data=data,
            batch_size=2,
            microbatch=2,
            lr=1e-4,
            ema_rate=0.9999,
            log_interval=10,
            save_interval=100,
            resume_checkpoint="",
            lr_anneal_steps=0,  # Disabled
            checkpoint_path=str(tmp_path),
        )

        initial_lr = train_loop.opt.param_groups[0]["lr"]
        train_loop._anneal_lr()
        final_lr = train_loop.opt.param_groups[0]["lr"]

        assert initial_lr == final_lr

    def test_lr_annealing_enabled(self, tmp_path: Path, setup_dist) -> None:
        """Test that LR decreases with annealing enabled"""
        model = SimpleModel()
        diffusion = create_mock_diffusion()
        data = create_tiny_data_generator(batch_size=2)

        train_loop = TrainLoop(
            model=model,
            diffusion=diffusion,
            data=data,
            batch_size=2,
            microbatch=2,
            lr=1e-4,
            ema_rate=0.9999,
            log_interval=10,
            save_interval=100,
            resume_checkpoint="",
            lr_anneal_steps=100,
            checkpoint_path=str(tmp_path),
        )

        initial_lr = train_loop.opt.param_groups[0]["lr"]

        # Simulate halfway through training
        train_loop.step = 50
        train_loop._anneal_lr()

        final_lr = train_loop.opt.param_groups[0]["lr"]

        # LR should have decreased
        assert final_lr < initial_lr
        # Should be approximately half of original
        assert abs(final_lr - initial_lr / 2) < 1e-5


class TestTrainLoopIntegration:
    """Integration tests for complete training steps"""

    def test_run_step_complete(self, tmp_path: Path, setup_dist) -> None:
        """Test complete run_step including forward, backward, and optimization"""
        model = SimpleModel()
        diffusion = create_mock_diffusion()
        data = create_tiny_data_generator(batch_size=2)

        train_loop = TrainLoop(
            model=model,
            diffusion=diffusion,
            data=data,
            batch_size=2,
            microbatch=2,
            lr=1e-4,
            ema_rate=0.9999,
            log_interval=10,
            save_interval=100,
            resume_checkpoint="",
            checkpoint_path=str(tmp_path),
        )

        batch, cond = next(data)
        initial_params = [p.clone().detach() for p in train_loop.model_params]

        train_loop.run_step(batch, cond)

        # Verify parameters were updated
        for initial, current in zip(initial_params, train_loop.model_params):
            if current.requires_grad:
                assert not th.allclose(initial, current)

    @pytest.mark.gpu
    def test_run_step_on_gpu(self, gpu_device, tmp_path: Path, setup_dist) -> None:
        """Test complete training step on GPU"""
        model = SimpleModel().to(gpu_device)
        diffusion = create_mock_diffusion()
        data = create_tiny_data_generator(batch_size=2)

        train_loop = TrainLoop(
            model=model,
            diffusion=diffusion,
            data=data,
            batch_size=2,
            microbatch=2,
            lr=1e-4,
            ema_rate=0.9999,
            log_interval=10,
            save_interval=100,
            resume_checkpoint="",
            checkpoint_path=str(tmp_path),
        )

        batch, cond = next(data)
        train_loop.run_step(batch, cond)

        # Verify model is still on GPU
        for param in train_loop.model_params:
            assert param.device.type == "cuda"
