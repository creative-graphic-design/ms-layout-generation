"""Comprehensive unit tests for transforms.py"""

from unittest.mock import Mock, patch

import numpy as np
import torch

from layoutprompter.transforms import (
    AddCanvasElement,
    AddGaussianNoise,
    AddRelation,
    CLIPTextEncoder,
    DiscretizeBoundingBox,
    LabelDictSort,
    LexicographicSort,
    RelationTypes,
    SaliencyMapToBBoxes,
    ShuffleElements,
)


class TestShuffleElements:
    """Test ShuffleElements transform"""

    def test_shuffle_elements(self):
        transform = ShuffleElements()
        data = {
            "labels": torch.tensor([0, 1, 2, 3]),
            "bboxes": torch.tensor(
                [
                    [0.1, 0.2, 0.3, 0.4],
                    [0.2, 0.3, 0.4, 0.5],
                    [0.3, 0.4, 0.5, 0.6],
                    [0.4, 0.5, 0.6, 0.7],
                ]
            ),
        }
        result = transform(data)
        assert "gold_bboxes" in result
        assert result["labels"].shape == torch.Size([4])
        assert result["bboxes"].shape == torch.Size([4, 4])
        # Check that all original labels are still present (just reordered)
        assert set(result["labels"].tolist()) == {0, 1, 2, 3}

    def test_shuffle_preserves_gold_bboxes(self):
        transform = ShuffleElements()
        data = {
            "labels": torch.tensor([0, 1, 2]),
            "bboxes": torch.tensor(
                [[0.1, 0.2, 0.3, 0.4], [0.2, 0.3, 0.4, 0.5], [0.3, 0.4, 0.5, 0.6]]
            ),
            "gold_bboxes": torch.tensor(
                [[0.5, 0.5, 0.5, 0.5], [0.6, 0.6, 0.6, 0.6], [0.7, 0.7, 0.7, 0.7]]
            ),
        }
        result = transform(data)
        # gold_bboxes should also be shuffled in the same order
        assert result["gold_bboxes"].shape == torch.Size([3, 4])

    def test_shuffle_single_element(self):
        transform = ShuffleElements()
        data = {
            "labels": torch.tensor([0]),
            "bboxes": torch.tensor([[0.1, 0.2, 0.3, 0.4]]),
        }
        result = transform(data)
        assert result["labels"].shape == torch.Size([1])


class TestLabelDictSort:
    """Test LabelDictSort transform"""

    def test_label_dict_sort(self, sample_id2label):
        transform = LabelDictSort(index2label=sample_id2label)
        data = {
            "labels": torch.tensor([2, 0, 1, 3]),  # button, text, image, icon
            "bboxes": torch.tensor(
                [
                    [0.1, 0.2, 0.3, 0.4],
                    [0.2, 0.3, 0.4, 0.5],
                    [0.3, 0.4, 0.5, 0.6],
                    [0.4, 0.5, 0.6, 0.7],
                ]
            ),
        }
        result = transform(data)
        # Should be sorted alphabetically: button(2), icon(3), image(1), text(0)
        expected_labels = [2, 3, 1, 0]
        assert result["labels"].tolist() == expected_labels

    def test_label_dict_sort_creates_gold_bboxes(self, sample_id2label):
        transform = LabelDictSort(index2label=sample_id2label)
        data = {
            "labels": torch.tensor([1, 0]),
            "bboxes": torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]),
        }
        result = transform(data)
        assert "gold_bboxes" in result
        assert result["gold_bboxes"].shape == result["bboxes"].shape

    def test_label_dict_sort_preserves_existing_gold_bboxes(self, sample_id2label):
        transform = LabelDictSort(index2label=sample_id2label)
        original_gold = torch.tensor([[0.9, 0.9, 0.9, 0.9], [0.8, 0.8, 0.8, 0.8]])
        data = {
            "labels": torch.tensor([1, 0]),
            "bboxes": torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]),
            "gold_bboxes": original_gold.clone(),
        }
        result = transform(data)
        # gold_bboxes should be reordered but values preserved
        assert result["gold_bboxes"].shape == original_gold.shape


class TestLexicographicSort:
    """Test LexicographicSort transform"""

    def test_lexicographic_sort_by_position(self):
        transform = LexicographicSort()
        data = {
            "labels": torch.tensor([0, 1, 2]),
            "bboxes": torch.tensor(
                [
                    [0.5, 0.6, 0.1, 0.1],
                    [0.3, 0.2, 0.1, 0.1],
                    [0.1, 0.2, 0.1, 0.1],
                ]  # bottom-right  # middle  # top-left
            ),
        }
        result = transform(data)
        # Should be sorted by (top, left): element 2, then 1, then 0
        assert result["labels"].tolist() == [2, 1, 0]

    def test_lexicographic_sort_creates_gold_bboxes(self):
        transform = LexicographicSort()
        data = {
            "labels": torch.tensor([0, 1]),
            "bboxes": torch.tensor([[0.5, 0.6, 0.1, 0.1], [0.1, 0.2, 0.1, 0.1]]),
        }
        result = transform(data)
        assert "gold_bboxes" in result
        assert "ori_bboxes" in result
        assert "ori_labels" in result

    def test_lexicographic_sort_same_top_position(self):
        transform = LexicographicSort()
        data = {
            "labels": torch.tensor([0, 1, 2]),
            "bboxes": torch.tensor(
                [
                    [0.5, 0.2, 0.1, 0.1],
                    [0.3, 0.2, 0.1, 0.1],
                    [0.1, 0.2, 0.1, 0.1],
                ]  # right  # middle  # left
            ),
        }
        result = transform(data)
        # Same top, sorted by left
        assert result["labels"].tolist() == [2, 1, 0]


class TestAddGaussianNoise:
    """Test AddGaussianNoise transform"""

    def test_add_gaussian_noise_normalized(self):
        transform = AddGaussianNoise(
            mean=0.0, std=0.01, normalized=True, bernoulli_beta=1.0
        )
        data = {
            "labels": torch.tensor([0, 1]),
            "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.2], [0.3, 0.3, 0.1, 0.1]]),
        }
        result = transform(data)
        assert "gold_bboxes" in result
        # Check that noise was applied (values should be different but close)
        assert not torch.allclose(result["bboxes"], result["gold_bboxes"])
        # Check clipping to [0, 1]
        assert torch.all(result["bboxes"] >= 0.0)
        assert torch.all(result["bboxes"] <= 1.0)

    def test_add_gaussian_noise_unnormalized(self):
        transform = AddGaussianNoise(
            mean=0.0, std=1.0, normalized=False, bernoulli_beta=1.0
        )
        data = {
            "labels": torch.tensor([0, 1]),
            "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.2], [0.3, 0.3, 0.1, 0.1]]),
            "canvas_size": [100, 100],
        }
        result = transform(data)
        assert "gold_bboxes" in result
        assert torch.all(result["bboxes"] >= 0.0)
        assert torch.all(result["bboxes"] <= 1.0)

    def test_add_gaussian_noise_bernoulli_beta_zero(self):
        transform = AddGaussianNoise(
            mean=0.0, std=0.1, normalized=True, bernoulli_beta=0.0
        )
        original_bboxes = torch.tensor([[0.5, 0.5, 0.2, 0.2], [0.3, 0.3, 0.1, 0.1]])
        data = {
            "labels": torch.tensor([0, 1]),
            "bboxes": original_bboxes.clone(),
        }
        result = transform(data)
        # With beta=0, no noise should be added
        assert torch.allclose(result["bboxes"], result["gold_bboxes"])

    def test_add_gaussian_noise_repr(self):
        transform = AddGaussianNoise(mean=0.5, std=0.1, bernoulli_beta=0.8)
        repr_str = repr(transform)
        assert "AddGaussianNoise" in repr_str
        assert "mean=0.5" in repr_str
        assert "std=0.1" in repr_str
        assert "beta=0.8" in repr_str


class TestDiscretizeBoundingBox:
    """Test DiscretizeBoundingBox transform"""

    def test_initialization(self):
        transform = DiscretizeBoundingBox(num_x_grid=100, num_y_grid=200)
        assert transform.num_x_grid == 100
        assert transform.num_y_grid == 200
        assert transform.max_x == 100
        assert transform.max_y == 200

    def test_discretize(self):
        transform = DiscretizeBoundingBox(num_x_grid=100, num_y_grid=100)
        bbox = torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.9, 1.0]])
        discrete = transform.discretize(bbox)
        assert discrete.dtype == torch.long
        assert discrete.shape == bbox.shape
        assert torch.all(discrete >= 0)
        assert torch.all(discrete[:, 0] <= 100)
        assert torch.all(discrete[:, 1] <= 100)

    def test_discretize_clips_values(self):
        transform = DiscretizeBoundingBox(num_x_grid=100, num_y_grid=100)
        # Values outside [0, 1] should be clipped
        bbox = torch.tensor([[-0.1, 0.5, 1.1, 0.8]])
        discrete = transform.discretize(bbox)
        # After clipping, x1=-0.1 becomes 0, x2=1.1 becomes 1.0
        assert discrete[0, 0] == 0
        assert discrete[0, 2] == 100

    def test_continuize(self):
        transform = DiscretizeBoundingBox(num_x_grid=100, num_y_grid=100)
        discrete_bbox = torch.tensor(
            [[10, 20, 30, 40], [50, 60, 90, 100]], dtype=torch.long
        )
        continuous = transform.continuize(discrete_bbox)
        assert continuous.dtype == torch.float
        assert continuous.shape == discrete_bbox.shape
        assert torch.all(continuous >= 0.0)
        assert torch.all(continuous <= 1.0)

    def test_continuize_num(self):
        transform = DiscretizeBoundingBox(num_x_grid=100, num_y_grid=100)
        assert transform.continuize_num(50) == 0.5
        assert transform.continuize_num(100) == 1.0
        assert transform.continuize_num(0) == 0.0

    def test_discretize_num(self):
        transform = DiscretizeBoundingBox(num_x_grid=100, num_y_grid=100)
        assert transform.discretize_num(0.5) == 50
        assert transform.discretize_num(1.0) == 100
        assert transform.discretize_num(0.0) == 0

    def test_call_creates_discrete_bboxes(self):
        transform = DiscretizeBoundingBox(num_x_grid=100, num_y_grid=100)
        data = {
            "labels": torch.tensor([0, 1]),
            "bboxes": torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]),
        }
        result = transform(data)
        assert "discrete_bboxes" in result
        assert "discrete_gold_bboxes" in result
        assert "gold_bboxes" in result
        assert result["discrete_bboxes"].dtype == torch.long

    def test_call_with_content_bboxes(self):
        transform = DiscretizeBoundingBox(num_x_grid=100, num_y_grid=100)
        data = {
            "labels": torch.tensor([0, 1]),
            "bboxes": torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]),
            "content_bboxes": torch.tensor([[0.2, 0.3, 0.4, 0.5]]),
        }
        result = transform(data)
        assert "discrete_content_bboxes" in result
        assert result["discrete_content_bboxes"].dtype == torch.long

    def test_discretize_continuize_roundtrip(self):
        transform = DiscretizeBoundingBox(num_x_grid=100, num_y_grid=100)
        original = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
        discrete = transform.discretize(original)
        recovered = transform.continuize(discrete)
        # Should be close but not exact due to floor operation
        assert torch.allclose(original, recovered, atol=0.01)


class TestAddCanvasElement:
    """Test AddCanvasElement transform"""

    def test_initialization(self):
        transform = AddCanvasElement(use_discrete=False)
        assert transform.use_discrete is False
        assert transform.x.shape == torch.Size([1, 4])
        assert transform.y.shape == torch.Size([1])
        assert torch.allclose(transform.x, torch.tensor([[0.0, 0.0, 1.0, 1.0]]))

    def test_call_without_discrete(self):
        transform = AddCanvasElement(use_discrete=False)
        data = {
            "labels": torch.tensor([1, 2]),
            "bboxes": torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]),
        }
        result = transform(data)
        assert "bboxes_with_canvas" in result
        assert "labels_with_canvas" in result
        # Canvas element should be first
        assert result["bboxes_with_canvas"].shape[0] == 3
        assert result["labels_with_canvas"].shape[0] == 3
        assert result["labels_with_canvas"][0] == 0

    def test_call_with_discrete(self):
        discrete_fn = DiscretizeBoundingBox(num_x_grid=100, num_y_grid=100)
        transform = AddCanvasElement(use_discrete=True, discrete_fn=discrete_fn)
        data = {
            "labels": torch.tensor([1, 2]),
            "bboxes": torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]),
            "discrete_gold_bboxes": torch.tensor(
                [[10, 20, 30, 40], [50, 60, 70, 80]], dtype=torch.long
            ),
        }
        result = transform(data)
        assert "bboxes_with_canvas" in result
        assert result["bboxes_with_canvas"].shape[0] == 3
        # First element should be canvas [0, 0, 1, 1]
        assert torch.allclose(
            result["bboxes_with_canvas"][0], torch.tensor([0.0, 0.0, 1.0, 1.0])
        )


class TestAddRelation:
    """Test AddRelation transform"""

    def test_initialization(self):
        transform = AddRelation(seed=42, ratio=0.1)
        assert transform.ratio == 0.1
        assert transform.type2index is not None

    def test_call_adds_relations(self):
        transform = AddRelation(seed=42, ratio=0.5)
        data = {
            "labels": torch.tensor([1, 2]),
            "labels_with_canvas": torch.tensor([0, 1, 2]),
            "bboxes_with_canvas": torch.tensor(
                [[0.0, 0.0, 1.0, 1.0], [0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.2, 0.3]]
            ),
        }
        result = transform(data)
        assert "relations" in result
        assert "labels_with_canvas_index" in result
        assert result["relations"].dtype == torch.long
        # Relations should have 5 columns: [label_i, idx_i, label_j, idx_j, rel_type]
        if result["relations"].numel() > 0:
            assert result["relations"].shape[1] == 5

    def test_call_with_small_ratio(self):
        transform = AddRelation(seed=42, ratio=0.01)
        data = {
            "labels": torch.tensor([1, 2]),
            "labels_with_canvas": torch.tensor([0, 1, 2]),
            "bboxes_with_canvas": torch.tensor(
                [[0.0, 0.0, 1.0, 1.0], [0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.2, 0.3]]
            ),
        }
        result = transform(data)
        assert "relations" in result
        # With small ratio, fewer relations should be generated


class TestRelationTypes:
    """Test RelationTypes class"""

    def test_types_list(self):
        assert RelationTypes.types == [
            "smaller",
            "equal",
            "larger",
            "top",
            "center",
            "bottom",
            "left",
            "right",
        ]

    def test_type2index(self):
        type2idx = RelationTypes.type2index()
        assert isinstance(type2idx, dict)
        assert type2idx["smaller"] == 0
        assert type2idx["top"] == 3
        assert len(type2idx) == 8

    def test_index2type(self):
        idx2type = RelationTypes.index2type()
        assert isinstance(idx2type, dict)
        assert idx2type[0] == "smaller"
        assert idx2type[3] == "top"
        assert len(idx2type) == 8

    def test_type2index_caching(self):
        # Call twice to ensure caching works
        first_call = RelationTypes.type2index()
        second_call = RelationTypes.type2index()
        assert first_call is second_call

    def test_index2type_caching(self):
        first_call = RelationTypes.index2type()
        second_call = RelationTypes.index2type()
        assert first_call is second_call


class TestSaliencyMapToBBoxes:
    """Test SaliencyMapToBBoxes transform"""

    def test_initialization(self):
        transform = SaliencyMapToBBoxes(threshold=100)
        assert transform.threshold == 100
        assert transform.is_filter_small_bboxes is True
        assert transform.min_side == 80
        assert transform.min_area == 6000

    def test_is_small_bbox(self):
        transform = SaliencyMapToBBoxes(threshold=100, min_side=50, min_area=5000)
        # Small by both dimensions
        assert transform._is_small_bbox([0, 0, 40, 40]) is True
        # Small by area
        assert transform._is_small_bbox([0, 0, 60, 60]) is True
        # Not small
        assert transform._is_small_bbox([0, 0, 100, 100]) is False

    def test_call_with_simple_saliency_map(self):
        transform = SaliencyMapToBBoxes(threshold=100, is_filter_small_bboxes=False)
        # Create a simple saliency map with a bright region
        saliency_map = np.zeros((200, 200, 3), dtype=np.uint8)
        saliency_map[50:150, 50:150] = 255
        bboxes = transform(saliency_map)
        assert isinstance(bboxes, torch.Tensor)
        assert bboxes.dim() == 2
        if bboxes.numel() > 0:
            assert bboxes.shape[1] == 4

    def test_call_with_no_bright_regions(self):
        transform = SaliencyMapToBBoxes(threshold=200)
        # Dark saliency map
        saliency_map = np.ones((100, 100, 3), dtype=np.uint8) * 50
        bboxes = transform(saliency_map)
        assert bboxes.shape[0] == 0

    def test_call_filters_small_bboxes(self):
        transform = SaliencyMapToBBoxes(
            threshold=100, is_filter_small_bboxes=True, min_side=50, min_area=5000
        )
        saliency_map = np.zeros((200, 200, 3), dtype=np.uint8)
        # Create small bright region (should be filtered)
        saliency_map[10:30, 10:30] = 255
        bboxes = transform(saliency_map)
        # Small bbox should be filtered out
        assert bboxes.shape[0] == 0

    def test_call_sorts_bboxes_by_position(self):
        transform = SaliencyMapToBBoxes(threshold=100, is_filter_small_bboxes=False)
        saliency_map = np.zeros((300, 300, 3), dtype=np.uint8)
        # Create multiple bright regions
        saliency_map[50:80, 50:80] = 255
        saliency_map[100:130, 50:80] = 255
        saliency_map[50:80, 150:180] = 255
        bboxes = transform(saliency_map)
        if bboxes.shape[0] > 1:
            # Check sorting by (y, x)
            for i in range(len(bboxes) - 1):
                y1, x1 = bboxes[i][1], bboxes[i][0]
                y2, x2 = bboxes[i + 1][1], bboxes[i + 1][0]
                assert (y1 < y2) or (y1 == y2 and x1 <= x2)


class TestCLIPTextEncoder:
    """Test CLIPTextEncoder transform"""

    @patch("layoutprompter.transforms.clip.load")
    def test_initialization(self, mock_load):
        mock_model = Mock()
        mock_preprocess = Mock()
        mock_load.return_value = (mock_model, mock_preprocess)
        encoder = CLIPTextEncoder(model_name="ViT-B/32")
        assert encoder.model_name == "ViT-B/32"
        mock_load.assert_called_once()

    @patch("layoutprompter.transforms.clip.load")
    @patch("layoutprompter.transforms.clip.tokenize")
    def test_call(self, mock_tokenize, mock_load):
        mock_model = Mock()
        mock_preprocess = Mock()
        mock_load.return_value = (mock_model, mock_preprocess)
        # Mock tokenize to return a tensor
        mock_tokenize.return_value = torch.tensor([[1, 2, 3]])
        # Mock encode_text to return embeddings
        mock_embedding = torch.randn(1, 512)
        mock_embedding = mock_embedding / mock_embedding.norm(dim=-1, keepdim=True)
        mock_model.encode_text.return_value = mock_embedding
        encoder = CLIPTextEncoder()
        result = encoder("test text")
        assert isinstance(result, torch.Tensor)
        mock_tokenize.assert_called_once()
        mock_model.encode_text.assert_called_once()

    @patch("layoutprompter.transforms.clip.load")
    def test_uses_cuda_when_available(self, mock_load):
        mock_model = Mock()
        mock_preprocess = Mock()
        mock_load.return_value = (mock_model, mock_preprocess)
        with patch("torch.cuda.is_available", return_value=True):
            encoder = CLIPTextEncoder()
            assert encoder.device == "cuda"

    @patch("layoutprompter.transforms.clip.load")
    def test_uses_cpu_when_cuda_unavailable(self, mock_load):
        mock_model = Mock()
        mock_preprocess = Mock()
        mock_load.return_value = (mock_model, mock_preprocess)
        with patch("torch.cuda.is_available", return_value=False):
            encoder = CLIPTextEncoder()
            assert encoder.device == "cpu"


class TestTransformCompositions:
    """Test compositions of multiple transforms"""

    def test_lexicographic_sort_then_discretize(self):
        sort_transform = LexicographicSort()
        discretize_transform = DiscretizeBoundingBox(num_x_grid=100, num_y_grid=100)
        data = {
            "labels": torch.tensor([0, 1, 2]),
            "bboxes": torch.tensor(
                [
                    [0.5, 0.6, 0.1, 0.1],
                    [0.1, 0.2, 0.1, 0.1],
                    [0.3, 0.4, 0.1, 0.1],
                ]  # bottom  # top  # middle
            ),
        }
        result = sort_transform(data)
        result = discretize_transform(result)
        # Should be sorted and discretized
        assert result["labels"].tolist() == [1, 2, 0]
        assert "discrete_bboxes" in result
        assert result["discrete_bboxes"].dtype == torch.long

    def test_shuffle_then_label_sort(self, sample_id2label):
        shuffle_transform = ShuffleElements()
        sort_transform = LabelDictSort(index2label=sample_id2label)
        data = {
            "labels": torch.tensor([2, 0, 1]),
            "bboxes": torch.tensor(
                [[0.1, 0.2, 0.3, 0.4], [0.2, 0.3, 0.4, 0.5], [0.3, 0.4, 0.5, 0.6]]
            ),
        }
        result = shuffle_transform(data)
        result = sort_transform(result)
        # After shuffling and sorting by label, should be in alphabetical order
        label_names = [sample_id2label[int(label_id)] for label_id in result["labels"]]
        assert label_names == sorted(label_names)

    def test_noise_then_discretize(self):
        noise_transform = AddGaussianNoise(mean=0.0, std=0.01, bernoulli_beta=1.0)
        discretize_transform = DiscretizeBoundingBox(num_x_grid=100, num_y_grid=100)
        data = {
            "labels": torch.tensor([0, 1]),
            "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.2], [0.3, 0.3, 0.1, 0.1]]),
        }
        result = noise_transform(data)
        result = discretize_transform(result)
        assert "discrete_bboxes" in result
        assert "gold_bboxes" in result
        assert "discrete_gold_bboxes" in result
