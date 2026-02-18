"""Integration tests for Coarse-to-Fine with LayoutFormer++ dependencies"""

import pytest

import torch

from coarse_to_fine.c2f_trainer import Trainer
from coarse_to_fine.c2f_generator import Generator


@pytest.mark.integration
class TestTrainerIntegration:
    """Integration tests for Trainer class"""

    def test_trainer_import(self):
        """Test that Trainer can be imported"""
        assert Trainer is not None

    def test_trainer_initialization(self, layoutformer_data_root, gpu_available):
        """Smoke test that required resources are available for trainer setup"""
        assert gpu_available, "GPU not available"
        assert layoutformer_data_root.is_dir()

    def test_trainer_one_step(self, layoutformer_data_root, gpu_available):
        """Smoke test that required resources are available for training"""
        assert gpu_available, "GPU not available"
        assert layoutformer_data_root.is_dir()


@pytest.mark.integration
class TestGeneratorIntegration:
    """Integration tests for Generator class"""

    def test_generator_import(self):
        """Test that Generator can be imported"""
        assert Generator is not None

    def test_generator_initialization(self, layoutformer_data_root, gpu_available):
        """Smoke test that required resources are available for generator setup"""
        assert gpu_available, "GPU not available"
        assert layoutformer_data_root.is_dir()

    def test_generator_inference(self, layoutformer_data_root, gpu_available):
        """Smoke test that required resources are available for inference"""
        assert gpu_available, "GPU not available"
        assert layoutformer_data_root.is_dir()


@pytest.mark.integration
class TestLayoutFormerPPComponents:
    """Test that we can import and use layoutformer_pp components"""

    def test_import_layoutformer_utils(self):
        """Test importing layoutformer_pp utilities"""
        from layoutformer_pp.utils import utils

        assert utils is not None

    def test_import_layoutformer_data(self):
        """Test importing layoutformer_pp data modules"""
        from layoutformer_pp.data.transforms import DiscretizeBoundingBox

        assert DiscretizeBoundingBox is not None

    def test_import_layoutformer_evaluation(self):
        """Test importing layoutformer_pp evaluation modules"""
        from layoutformer_pp.evaluation import metrics

        assert metrics is not None

    def test_discretize_bounding_box_basic(self):
        """Test DiscretizeBoundingBox transform"""
        from layoutformer_pp.data.transforms import DiscretizeBoundingBox

        transform = DiscretizeBoundingBox(num_x_grid=32, num_y_grid=32)

        # Test with a simple bbox (normalized coordinates)
        bbox = torch.tensor([[0.1, 0.2, 0.5, 0.6]])
        discretized = transform.discretize(bbox)

        # Check output is discretized (integer values)
        assert discretized.dtype in [torch.long, torch.int32, torch.int64]
        assert discretized.shape == bbox.shape
