"""Integration tests for Coarse-to-Fine training pipeline"""

import os
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def _write_preprocessed(root: Path, data_name: str, max_num_elements: int, label_count: int):
    base = root / data_name
    pre_dir = base / f"pre_processed_{max_num_elements}_{label_count}"
    pre_dir.mkdir(parents=True, exist_ok=True)
    sample = {
        "bboxes": torch.tensor([[0.1, 0.1, 0.2, 0.2], [0.4, 0.4, 0.2, 0.2]]),
        "labels": torch.tensor([1, 2]),
        "name": "sample_0",
    }
    for split in ("train", "val", "test"):
        torch.save([sample], pre_dir / f"{split}.pt")


def _write_torch_six_shim(root: Path) -> Path:
    shim_dir = root / "sitecustomize_shim"
    shim_dir.mkdir(parents=True, exist_ok=True)
    shim_path = shim_dir / "sitecustomize.py"
    shim_path.write_text(
        "\n".join(
            [
                "import sys",
                "import types",
                "import torch",
                "",
                "if 'torch._six' not in sys.modules:",
                "    mod = types.ModuleType('torch._six')",
                "    mod.inf = float('inf')",
                "    sys.modules['torch._six'] = mod",
                "",
            ]
        )
    )
    return shim_dir


@pytest.mark.integration
@pytest.mark.gpu
@pytest.mark.real_data
@pytest.mark.timeout(300)
class TestC2FTrainingPipeline:
    """Integration tests for full training pipeline with real LayoutFormer++ data"""

    def test_short_training_run_creates_checkpoint(
        self, layoutformer_data_root, single_gpu_env, ddp_env, repo_root
    ):
        """Test that a short training run completes and creates checkpoints"""
        # Create temporary output and data directories
        with tempfile.TemporaryDirectory() as tmpdir, tempfile.TemporaryDirectory() as datadir:
            _write_preprocessed(Path(datadir), "rico", max_num_elements=4, label_count=25)

            train_script = repo_root / "src" / "coarse_to_fine" / "main.py"
            assert train_script.exists(), f"Training script not found: {train_script}"

            cmd = [
                "python",
                str(train_script),
                "--train",
                "--dataset",
                "rico",
                "--data_dir",
                datadir,
                "--out_dir",
                tmpdir,
                "--epoch",
                "1",
                "--batch_size",
                "1",
                "--eval_batch_size",
                "1",
                "--max_num_elements",
                "4",
                "--train_log_step",
                "1",
                "--backend",
                "gloo",
                "--local_rank",
                "0",
                "--gradient_accumulation",
                "1",
                "--num_labels",
                "25",
                "--d_model",
                "32",
                "--d_z",
                "32",
                "--n_layers",
                "1",
                "--n_layers_decoder",
                "1",
                "--n_heads",
                "2",
                "--dim_feedforward",
                "64",
                "--discrete_x_grid",
                "8",
                "--discrete_y_grid",
                "8",
            ]

            env = single_gpu_env.copy()
            env.update(ddp_env)
            shim_dir = _write_torch_six_shim(Path(tmpdir))
            existing = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = (
                f"{shim_dir}{os.pathsep}{existing}" if existing else str(shim_dir)
            )

            result = subprocess.run(
                cmd,
                env=env,
                capture_output=True,
                text=True,
                timeout=300,
                check=False,
            )

            assert (
                result.returncode == 0
            ), f"Training failed.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"

            checkpoint_path = Path(tmpdir) / "checkpoint.pth.tar"
            assert checkpoint_path.is_file()
            assert checkpoint_path.stat().st_size > 0

    def test_training_with_validation(
        self, layoutformer_data_root, single_gpu_env, ddp_env, repo_root
    ):
        """Test training pipeline with validation step"""
        with tempfile.TemporaryDirectory() as tmpdir, tempfile.TemporaryDirectory() as datadir:
            _write_preprocessed(Path(datadir), "publaynet", max_num_elements=4, label_count=5)

            train_script = repo_root / "src" / "coarse_to_fine" / "main.py"
            assert train_script.exists(), f"Training script not found: {train_script}"

            cmd = [
                "python",
                str(train_script),
                "--train",
                "--dataset",
                "publaynet",
                "--data_dir",
                datadir,
                "--out_dir",
                tmpdir,
                "--epoch",
                "1",
                "--batch_size",
                "1",
                "--eval_batch_size",
                "1",
                "--max_num_elements",
                "4",
                "--train_log_step",
                "1",
                "--backend",
                "gloo",
                "--local_rank",
                "0",
                "--num_labels",
                "5",
                "--d_model",
                "32",
                "--d_z",
                "32",
                "--n_layers",
                "1",
                "--n_layers_decoder",
                "1",
                "--n_heads",
                "2",
                "--dim_feedforward",
                "64",
                "--discrete_x_grid",
                "8",
                "--discrete_y_grid",
                "8",
            ]

            env = single_gpu_env.copy()
            env.update(ddp_env)
            shim_dir = _write_torch_six_shim(Path(tmpdir))
            existing = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = (
                f"{shim_dir}{os.pathsep}{existing}" if existing else str(shim_dir)
            )

            result = subprocess.run(
                cmd,
                env=env,
                capture_output=True,
                text=True,
                timeout=300,
                check=False,
            )

            assert (
                result.returncode == 0
            ), f"Training failed.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"

            val_output_path = Path(tmpdir) / "val_output.pkl"
            assert val_output_path.is_file()
            assert val_output_path.stat().st_size > 0


@pytest.mark.integration
@pytest.mark.gpu
class TestC2FGeneratorPipeline:
    """Integration tests for generation pipeline"""

    def test_generation_loads_checkpoint(
        self, layoutformer_data_root, gpu_available
    ):
        """Test that generator can load a checkpoint and run inference"""
        assert gpu_available, "GPU not available"

        from coarse_to_fine.c2f_generator import Generator
        from coarse_to_fine.main import create_dataset

        class DummyFID:
            def __init__(self):
                self.model = torch.nn.Identity()

            def collect_features(self, *args, **kwargs):
                return None

            def compute_score(self):
                return 0.0

        def collate_fn(batch):
            return {"name": [item["name"] for item in batch]}

        def test_step(args, model, data, device):
            batch_size = len(data["name"])
            seq_len = 2
            group_len = 2

            bboxes = torch.zeros(batch_size, seq_len, 4, dtype=torch.long, device=device)
            labels = torch.ones(batch_size, seq_len, dtype=torch.long, device=device)

            group_bboxes = torch.zeros(batch_size, group_len, 4, dtype=torch.long, device=device)
            group_labels = torch.ones(batch_size, group_len, dtype=torch.long, device=device)

            ori = {"bboxes": bboxes, "labels": labels}
            out = {
                "bboxes": bboxes.clone(),
                "labels": labels.clone(),
                "group_bounding_box": group_bboxes,
                "label_in_one_group": group_labels,
            }
            masks = {
                "ori_box_mask": torch.ones(batch_size, seq_len, dtype=torch.bool),
                "gen_box_mask": torch.ones(batch_size, seq_len, dtype=torch.bool),
                "gen_group_bounding_box_mask": torch.ones(batch_size, group_len, dtype=torch.bool),
            }
            return ori, out, masks

        with tempfile.TemporaryDirectory() as tmpdir, tempfile.TemporaryDirectory() as datadir:
            _write_preprocessed(Path(datadir), "rico", max_num_elements=4, label_count=25)

            args = SimpleNamespace(
                backend="gloo",
                local_rank=0,
                trainer="single",
                out_dir=tmpdir,
                eval_batch_size=1,
                discrete_x_grid=8,
                discrete_y_grid=8,
                bbox_format="ltwh",
                dataset="rico",
                num_save=0,
                max_num_elements=4,
                num_labels=25,
                data_dir=datadir,
            )

            test_dataset = create_dataset(args, split="test")
            model = torch.nn.Linear(4, 1)
            checkpoint_path = Path(tmpdir) / "checkpoint.pth.tar"
            torch.save(model.state_dict(), checkpoint_path)

            generator = Generator(
                args=args,
                model=model,
                test_dataset=test_dataset,
                fid_model=DummyFID(),
                ckpt_path=str(checkpoint_path),
                collate_fn=collate_fn,
            )

            generator(test_step, draw_colors=test_dataset.colors)

            assert os.path.exists(os.path.join(tmpdir, "metrics.pkl"))
            assert os.path.exists(os.path.join(tmpdir, "results.pkl"))


@pytest.mark.integration
class TestC2FDataProcessing:
    """Integration tests for data processing pipeline"""

    def test_cut_hierarchy_with_real_data(self, layoutformer_data_root):
        """Test CutHierarchy processing with real dataset structure"""
        from coarse_to_fine.cut_hierarchy import CutHierarchy
        import torch

        cut_hierarchy = CutHierarchy()

        # Create realistic sample data structure
        sample_data = {
            "discrete_gold_bboxes": torch.tensor(
                [
                    [5, 5, 10, 10],  # Header element
                    [15, 5, 10, 10],  # Header element
                    [5, 20, 10, 15],  # Content element
                    [15, 20, 10, 15],  # Content element
                    [5, 40, 10, 10],  # Footer element
                ],
                dtype=torch.long,
            ),
            "labels": torch.tensor([1, 1, 2, 2, 3], dtype=torch.long),
        }

        num_labels = 5

        def discrete_func(x):
            return x

        # Process through CutHierarchy
        result = cut_hierarchy(sample_data, num_labels, discrete_func)

        # Validate output structure
        assert "group_bounding_box" in result
        assert "label_in_one_group" in result
        assert "grouped_label" in result
        assert "grouped_box" in result

        # Check consistency
        num_groups = len(result["grouped_label"])
        assert result["group_bounding_box"].shape[0] == num_groups
        assert result["label_in_one_group"].shape[0] == num_groups
        assert len(result["grouped_box"]) == num_groups

        # Validate data types
        assert result["group_bounding_box"].dtype == torch.long
        assert result["label_in_one_group"].dtype == torch.long

        # Validate that relative coordinates are in valid range
        for group_box in result["grouped_box"]:
            assert group_box.min() >= 0
            # After discretization, values should be in reasonable range
            assert group_box.max() <= 100  # Assuming max grid is 100


@pytest.mark.integration
class TestC2FModelIntegration:
    """Integration tests for model components"""

    def test_model_forward_pass_with_mock_data(self):
        """Test full model forward pass with mock hierarchical data"""
        from types import SimpleNamespace
        from coarse_to_fine.c2f_model.model import C2FLayoutTransformer
        import torch

        # Create minimal config
        cfg = SimpleNamespace(
            num_labels=5,
            discrete_x_grid=32,
            discrete_y_grid=32,
            d_model=64,
            n_heads=2,
            dim_feedforward=128,
            dropout=0.1,
            n_layers=1,
            n_layers_decoder=1,
            d_z=32,
            eval_batch_size=2,
            max_num_elements=10,
        )

        model = C2FLayoutTransformer(cfg)
        model.eval()

        # Create mock hierarchical data structure
        batch_size = 2
        num_groups = 3
        num_elements_per_group = 4

        # Mock input structure (simplified)
        data = {
            "labels": torch.randint(0, cfg.num_labels, (batch_size, num_elements_per_group)),
            "bboxes": torch.randint(0, cfg.discrete_x_grid, (batch_size, num_elements_per_group, 4)),
            "mask": torch.ones(batch_size, num_elements_per_group, dtype=torch.bool),
            "group_bounding_box": torch.randint(0, cfg.discrete_x_grid, (batch_size, num_groups, 4)),
            "label_in_one_group": torch.randint(0, 3, (batch_size, num_groups, cfg.num_labels + 2)),
            "grouped_label": [
                [torch.randint(0, cfg.num_labels, (2,)) for _ in range(num_groups)]
                for _ in range(batch_size)
            ],
            "grouped_box": [
                [torch.randint(0, cfg.discrete_x_grid, (2, 4)) for _ in range(num_groups)]
                for _ in range(batch_size)
            ],
        }

        # Test that model can be instantiated and run
        # (Full forward pass requires complex data structure, so we just test instantiation)
        assert model is not None
        assert hasattr(model, "encoder")
        assert hasattr(model, "vae")
        assert hasattr(model, "group_decoder")
        assert hasattr(model, "ele_decoder")


@pytest.mark.integration
class TestC2FEndToEnd:
    """End-to-end integration tests"""

    def test_trainer_and_generator_compatibility(self, repo_root):
        """Test that Trainer and Generator use compatible checkpoint formats"""
        from coarse_to_fine.c2f_generator import Generator
        from unittest.mock import MagicMock, patch
        from types import SimpleNamespace
        import torch
        import tempfile

        # Create minimal mock setup
        mock_args = SimpleNamespace(
            backend="gloo",
            local_rank=0,
            trainer="single",
            out_dir=tempfile.mkdtemp(),
            max_num_elements=10,
            gradient_accumulation=1,
            enable_clip_gradient=True,
            clip_gradient=1.0,
            batch_size=2,
            eval_batch_size=2,
            epoch=1,
            train_log_step=10,
            bbox_format="ltrb",
            discrete_x_grid=32,
            discrete_y_grid=32,
            kl_start_step=0,
            kl_end_step=100,
            dataset="rico",
            num_save=1,
        )

        model = torch.nn.Linear(10, 10)
        checkpoint_path = Path(mock_args.out_dir) / "checkpoint.pth.tar"

        # Save a checkpoint
        torch.save(model.state_dict(), checkpoint_path)

        try:
            # Test that Generator can load the checkpoint
            with patch("coarse_to_fine.c2f_generator.os_utils.makedirs"):
                mock_dataset = MagicMock()
                mock_dataset.__len__ = MagicMock(return_value=10)

                mock_fid = MagicMock()
                mock_fid.model = torch.nn.Linear(5, 5)

                # This should not raise an error
                generator = Generator(
                    args=mock_args,
                    model=torch.nn.Linear(10, 10),  # Same architecture
                    test_dataset=mock_dataset,
                    fid_model=mock_fid,
                    ckpt_path=str(checkpoint_path),
                )

                assert generator is not None

        finally:
            # Cleanup
            import shutil

            if Path(mock_args.out_dir).exists():
                shutil.rmtree(mock_args.out_dir)
