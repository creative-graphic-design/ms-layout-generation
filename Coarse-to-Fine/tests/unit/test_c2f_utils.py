"""Unit tests for c2f_utils.py functions"""

import torch
import pytest

from coarse_to_fine.c2f_utils import to_dense_batch, padding, get_mask, repeat_unit, cal_loss


class TestToDenseBatch:
    """Tests for to_dense_batch function"""

    def test_basic_2d_tensors(self):
        """Test padding 2D tensors to dense batch"""
        batch = [torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([[5.0, 6.0]])]
        out, mask = to_dense_batch(batch)

        assert out.shape == (2, 2, 2)
        assert mask.shape == (2, 2)
        assert mask.dtype == torch.bool

        # Check first batch item (no padding)
        assert torch.equal(out[0, 0], batch[0][0])
        assert torch.equal(out[0, 1], batch[0][1])

        # Check second batch item (with padding)
        assert torch.equal(out[1, 0], batch[1][0])
        assert torch.equal(out[1, 1], torch.zeros(2))

        # Check masks
        assert mask[0].tolist() == [True, True]
        assert mask[1].tolist() == [True, False]

    def test_1d_tensors(self):
        """Test padding 1D tensors (auto unsqueeze)"""
        batch = [torch.tensor([1, 2, 3]), torch.tensor([4, 5])]
        out, mask = to_dense_batch(batch)

        assert out.shape == (2, 3)
        assert mask.shape == (2, 3)

        # Check masks
        assert mask[0].tolist() == [True, True, True]
        assert mask[1].tolist() == [True, True, False]

    def test_custom_max_lens(self):
        """Test with custom max_lens parameter"""
        batch = [torch.tensor([[1.0, 2.0]]), torch.tensor([[3.0, 4.0]])]
        out, mask = to_dense_batch(batch, max_lens=3)

        assert out.shape == (2, 3, 2)
        assert mask.shape == (2, 3)

        # Both should have 2 True values and 1 False
        assert mask[0].tolist() == [True, False, False]
        assert mask[1].tolist() == [True, False, False]


class TestPadding:
    """Tests for padding function"""

    @pytest.fixture
    def sample_data(self):
        """Create sample input data for padding function"""
        return {
            "labels": [torch.tensor([1, 2]), torch.tensor([3])],
            "bboxes": [
                torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]),
                torch.tensor([[0.9, 1.0, 1.1, 1.2]]),
            ],
            "label_in_one_group": [
                torch.tensor([0.0, 1.0, 0.0]),
                torch.tensor([1.0, 0.0, 0.0]),
            ],
            "group_bounding_box": [
                torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
                torch.tensor([[0.0, 0.0, 0.5, 0.5]]),
            ],
            "grouped_label": [
                [torch.tensor([1]), torch.tensor([2, 3])],
                [torch.tensor([4])],
            ],
            "grouped_box": [
                [
                    torch.tensor([[0.1, 0.1, 0.2, 0.2]]),
                    torch.tensor([[0.3, 0.3, 0.4, 0.4], [0.5, 0.5, 0.6, 0.6]]),
                ],
                [torch.tensor([[0.7, 0.7, 0.8, 0.8]])],
            ],
        }

    def test_padding_structure(self, sample_data):
        """Test that padding returns correct structure"""
        device = torch.device("cpu")
        result = padding(sample_data, device)

        # Check all expected keys are present
        expected_keys = [
            "bboxes",
            "labels",
            "masks",
            "group_bounding_box",
            "label_in_one_group",
            "group_masks",
            "grouped_bboxes",
            "grouped_labels",
            "grouped_ele_masks",
        ]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"
            assert isinstance(result[key], torch.Tensor), (
                f"Key {key} should be a tensor"
            )

    def test_padding_shapes(self, sample_data):
        """Test that padded tensors have correct shapes"""
        device = torch.device("cpu")
        result = padding(sample_data, device)

        # Check batch size is consistent
        batch_size = result["labels"].shape[0]
        assert batch_size == 2

        # Check all tensors are on the correct device
        for key in ["bboxes", "labels", "masks"]:
            assert result[key].device == device


class TestGetMask:
    """Tests for get_mask function"""

    @pytest.fixture
    def sample_ori(self):
        """Create sample padded data for get_mask function"""
        return {
            "masks": torch.tensor([[True, True, False], [True, False, False]]),
            "group_masks": torch.tensor([[True, True], [True, False]]),
            "grouped_ele_masks": torch.tensor(
                [
                    [[True, True, False], [True, False, False]],
                    [[True, False, False], [True, True, False]],
                ]
            ),
        }

    def test_get_mask_structure(self, sample_ori):
        """Test that get_mask returns correct structure"""
        device = torch.device("cpu")
        result = get_mask(sample_ori, device)

        # Check expected keys
        expected_keys = [
            "ori_box_mask",
            "rec_label_in_one_group_mask",
            "ori_label_in_one_group_mask",
            "rec_group_bounding_box_mask",
            "ori_group_bounding_box_mask",
            "rec_grouped_label_mask",
            "ori_grouped_label_mask",
            "rec_grouped_box_mask",
            "ori_grouped_box_mask",
        ]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"
            assert isinstance(result[key], torch.Tensor), (
                f"Key {key} should be a tensor"
            )

    def test_get_mask_shapes(self, sample_ori):
        """Test that mask tensors have correct shapes"""
        device = torch.device("cpu")
        result = get_mask(sample_ori, device)

        # Check 2D masks
        assert result["ori_box_mask"].shape == sample_ori["masks"].shape
        assert (
            result["ori_label_in_one_group_mask"].shape
            == sample_ori["group_masks"].shape
        )

        # Check 3D masks
        assert (
            result["ori_grouped_label_mask"].shape
            == sample_ori["grouped_ele_masks"].shape
        )
        assert (
            result["ori_grouped_box_mask"].shape
            == sample_ori["grouped_ele_masks"].shape
        )


class TestRepeatUnitAndCalLoss:
    def test_repeat_unit_unsupported_dim(self):
        fill_unit = torch.zeros(1, 1, 1)
        with pytest.raises(NotImplementedError):
            repeat_unit(fill_unit, max_lens=3, cur_len=1)

    def test_cal_loss_returns_all_keys(self):
        device = torch.device("cpu")
        n, g, s = 1, 2, 3
        num_labels = 3
        ori = {
            "bboxes": torch.zeros(n, s, 4),
            "labels": torch.zeros(n, s, dtype=torch.long),
            "masks": torch.ones(n, s, dtype=torch.bool),
            "group_bounding_box": torch.zeros(n, g + 2, 4),
            "label_in_one_group": torch.zeros(n, g + 2, num_labels + 2),
            "group_masks": torch.ones(n, g + 2, dtype=torch.bool),
            "grouped_bboxes": torch.zeros(n, g, s, 4),
            "grouped_labels": torch.zeros(n, g, s),
            "grouped_ele_masks": torch.ones(n, g, s, dtype=torch.bool),
        }

        class Args:
            num_labels = 3
            discrete_x_grid = 8
            discrete_y_grid = 8
            group_box_weight = 1.0
            group_label_weight = 1.0
            box_weight = 1.0
            label_weight = 1.0
            kl_weight = 1.0

        d_box = max(Args.discrete_x_grid, Args.discrete_y_grid)
        n, g, s, _ = ori["grouped_bboxes"].shape
        rec = {
            "grouped_bboxes": torch.randn(n, g, s, 4, d_box),
            "grouped_labels": torch.randn(n, g, s, Args.num_labels + 3),
            "group_bounding_box": torch.randn(n, g + 2, 4, d_box),
            "label_in_one_group": torch.randn(n, g + 2, Args.num_labels + 2),
        }
        kl_info = {"mu": torch.zeros(1, 1), "logvar": torch.zeros(1, 1)}

        loss = cal_loss(Args, ori, rec, kl_info, device)
        assert set(loss.keys()) == {
            "group_bounding_box",
            "label_in_one_group",
            "grouped_box",
            "grouped_label",
            "KL",
        }
