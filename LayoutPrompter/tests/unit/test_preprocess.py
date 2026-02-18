"""Comprehensive unit tests for preprocess.py"""

from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch
from pandas import DataFrame

from layoutprompter.preprocess import (
    CompletionProcessor,
    ContentAwareProcessor,
    GenRelationProcessor,
    GenTypeProcessor,
    GenTypeSizeProcessor,
    Processor,
    RefinementProcessor,
    TextToLayoutProcessor,
    create_processor,
)


class TestProcessor:
    """Test base Processor class"""

    def test_initialization_with_sort_by_pos(self, sample_id2label, canvas_size):
        processor = Processor(
            index2label=sample_id2label,
            canvas_width=canvas_size[0],
            canvas_height=canvas_size[1],
            sort_by_pos=True,
        )
        assert processor.index2label == sample_id2label
        assert processor.canvas_width == canvas_size[0]
        assert processor.canvas_height == canvas_size[1]
        assert processor.sort_by_pos is True
        assert len(processor.transform_functions) == 2

    def test_initialization_with_shuffle_before_sort_by_label(
        self, sample_id2label, canvas_size
    ):
        processor = Processor(
            index2label=sample_id2label,
            canvas_width=canvas_size[0],
            canvas_height=canvas_size[1],
            shuffle_before_sort_by_label=True,
        )
        assert processor.shuffle_before_sort_by_label is True
        assert len(processor.transform_functions) == 3

    def test_initialization_with_sort_by_pos_before_sort_by_label(
        self, sample_id2label, canvas_size
    ):
        processor = Processor(
            index2label=sample_id2label,
            canvas_width=canvas_size[0],
            canvas_height=canvas_size[1],
            sort_by_pos_before_sort_by_label=True,
        )
        assert processor.sort_by_pos_before_sort_by_label is True
        assert len(processor.transform_functions) == 3

    def test_initialization_without_sort_options_raises_error(
        self, sample_id2label, canvas_size
    ):
        with pytest.raises(ValueError, match="At least one of"):
            Processor(
                index2label=sample_id2label,
                canvas_width=canvas_size[0],
                canvas_height=canvas_size[1],
                sort_by_pos=False,
                shuffle_before_sort_by_label=False,
                sort_by_pos_before_sort_by_label=False,
            )

    def test_config_base_transform_sort_by_pos(self, sample_id2label, canvas_size):
        processor = Processor(
            index2label=sample_id2label,
            canvas_width=canvas_size[0],
            canvas_height=canvas_size[1],
            sort_by_pos=True,
        )
        transforms = processor.transform_functions
        assert len(transforms) == 2
        assert transforms[0].__class__.__name__ == "LexicographicSort"
        assert transforms[1].__class__.__name__ == "DiscretizeBoundingBox"

    def test_call_returns_correct_keys(self, sample_id2label, canvas_size):
        # Create a concrete subclass for testing
        class TestProcessorSubclass(Processor):
            return_keys = ["labels", "bboxes"]

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.transform = lambda x: x

        processor = TestProcessorSubclass(
            index2label=sample_id2label,
            canvas_width=canvas_size[0],
            canvas_height=canvas_size[1],
            sort_by_pos=True,
        )
        data = {
            "labels": torch.tensor([0, 1]),
            "bboxes": torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]),
            "extra_key": "should_be_filtered",
        }
        result = processor(data)
        assert set(result.keys()) == {"labels", "bboxes"}
        assert "extra_key" not in result


class TestGenTypeProcessor:
    """Test GenTypeProcessor class"""

    def test_initialization(self, sample_id2label, canvas_size):
        processor = GenTypeProcessor(
            index2label=sample_id2label,
            canvas_width=canvas_size[0],
            canvas_height=canvas_size[1],
        )
        assert processor.index2label == sample_id2label
        assert processor.canvas_width == canvas_size[0]
        assert processor.canvas_height == canvas_size[1]

    def test_return_keys(self):
        assert GenTypeProcessor.return_keys == [
            "labels",
            "bboxes",
            "gold_bboxes",
            "discrete_bboxes",
            "discrete_gold_bboxes",
        ]

    def test_process_with_default_params(self, sample_id2label, canvas_size):
        processor = GenTypeProcessor(
            index2label=sample_id2label,
            canvas_width=canvas_size[0],
            canvas_height=canvas_size[1],
        )
        data = {
            "labels": torch.tensor([0, 1, 2]),
            "bboxes": torch.tensor(
                [[0.1, 0.2, 0.3, 0.4], [0.5, 0.1, 0.2, 0.3], [0.2, 0.3, 0.4, 0.5]]
            ),
        }
        result = processor(data)
        assert "labels" in result
        assert "bboxes" in result
        assert "gold_bboxes" in result
        assert "discrete_bboxes" in result
        assert "discrete_gold_bboxes" in result
        assert result["labels"].shape[0] == 3
        assert result["discrete_bboxes"].dtype == torch.long

    def test_process_with_sort_by_pos(self, sample_id2label, canvas_size):
        processor = GenTypeProcessor(
            index2label=sample_id2label,
            canvas_width=canvas_size[0],
            canvas_height=canvas_size[1],
            sort_by_pos=True,
        )
        data = {
            "labels": torch.tensor([0, 1, 2]),
            "bboxes": torch.tensor(
                [[0.5, 0.6, 0.1, 0.1], [0.1, 0.2, 0.1, 0.1], [0.3, 0.4, 0.1, 0.1]]
            ),
        }
        result = processor(data)
        # Elements should be sorted by position (top, then left)
        assert result["labels"][0] == 1  # First by top position


class TestGenTypeSizeProcessor:
    """Test GenTypeSizeProcessor class"""

    def test_initialization(self, sample_id2label, canvas_size):
        processor = GenTypeSizeProcessor(
            index2label=sample_id2label,
            canvas_width=canvas_size[0],
            canvas_height=canvas_size[1],
        )
        assert processor.shuffle_before_sort_by_label is True

    def test_return_keys(self):
        assert GenTypeSizeProcessor.return_keys == [
            "labels",
            "bboxes",
            "gold_bboxes",
            "discrete_bboxes",
            "discrete_gold_bboxes",
        ]

    def test_process_with_shuffle(self, sample_id2label, canvas_size):
        processor = GenTypeSizeProcessor(
            index2label=sample_id2label,
            canvas_width=canvas_size[0],
            canvas_height=canvas_size[1],
        )
        data = {
            "labels": torch.tensor([0, 1, 2]),
            "bboxes": torch.tensor(
                [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8], [0.2, 0.3, 0.4, 0.5]]
            ),
        }
        result = processor(data)
        assert "labels" in result
        assert result["labels"].shape[0] == 3
        assert "discrete_bboxes" in result


class TestGenRelationProcessor:
    """Test GenRelationProcessor class"""

    def test_initialization(self, sample_id2label, canvas_size):
        processor = GenRelationProcessor(
            index2label=sample_id2label,
            canvas_width=canvas_size[0],
            canvas_height=canvas_size[1],
        )
        assert processor.index2label == sample_id2label

    def test_return_keys(self):
        assert GenRelationProcessor.return_keys == [
            "labels",
            "bboxes",
            "gold_bboxes",
            "discrete_bboxes",
            "discrete_gold_bboxes",
            "relations",
        ]

    def test_process_without_relation_constraint(self, sample_id2label, canvas_size):
        processor = GenRelationProcessor(
            index2label=sample_id2label,
            canvas_width=canvas_size[0],
            canvas_height=canvas_size[1],
            relation_constrained_discrete_before_induce_relations=False,
        )
        data = {
            "labels": torch.tensor([0, 1]),
            "bboxes": torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.2, 0.3]]),
        }
        result = processor(data)
        assert "relations" in result
        if result["relations"].numel() > 0:
            assert (
                result["relations"].shape[1] == 5
            )  # [label_i, idx_i, label_j, idx_j, rel_type]

    def test_process_with_relation_constraint(self, sample_id2label, canvas_size):
        processor = GenRelationProcessor(
            index2label=sample_id2label,
            canvas_width=canvas_size[0],
            canvas_height=canvas_size[1],
            relation_constrained_discrete_before_induce_relations=True,
        )
        data = {
            "labels": torch.tensor([0, 1]),
            "bboxes": torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.2, 0.3]]),
        }
        result = processor(data)
        assert "relations" in result
        assert "discrete_bboxes" in result


class TestCompletionProcessor:
    """Test CompletionProcessor class"""

    def test_initialization(self, sample_id2label, canvas_size):
        processor = CompletionProcessor(
            index2label=sample_id2label,
            canvas_width=canvas_size[0],
            canvas_height=canvas_size[1],
        )
        assert processor.sort_by_pos is True

    def test_return_keys(self):
        assert CompletionProcessor.return_keys == [
            "labels",
            "bboxes",
            "gold_bboxes",
            "discrete_bboxes",
            "discrete_gold_bboxes",
        ]

    def test_process(self, sample_id2label, canvas_size):
        processor = CompletionProcessor(
            index2label=sample_id2label,
            canvas_width=canvas_size[0],
            canvas_height=canvas_size[1],
        )
        data = {
            "labels": torch.tensor([0, 1, 2]),
            "bboxes": torch.tensor(
                [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8], [0.2, 0.3, 0.4, 0.5]]
            ),
        }
        result = processor(data)
        assert len(result["labels"]) == 3


class TestRefinementProcessor:
    """Test RefinementProcessor class"""

    def test_initialization(self, sample_id2label, canvas_size):
        processor = RefinementProcessor(
            index2label=sample_id2label,
            canvas_width=canvas_size[0],
            canvas_height=canvas_size[1],
        )
        assert processor.index2label == sample_id2label

    def test_initialization_with_noise_params(self, sample_id2label, canvas_size):
        processor = RefinementProcessor(
            index2label=sample_id2label,
            canvas_width=canvas_size[0],
            canvas_height=canvas_size[1],
            gaussian_noise_mean=0.1,
            gaussian_noise_std=0.05,
            train_bernoulli_beta=0.8,
        )
        # Check that noise transform is added first
        assert processor.transform_functions[0].__class__.__name__ == "AddGaussianNoise"

    def test_return_keys(self):
        assert RefinementProcessor.return_keys == [
            "labels",
            "bboxes",
            "gold_bboxes",
            "discrete_bboxes",
            "discrete_gold_bboxes",
        ]

    def test_process_adds_noise(self, sample_id2label, canvas_size):
        processor = RefinementProcessor(
            index2label=sample_id2label,
            canvas_width=canvas_size[0],
            canvas_height=canvas_size[1],
            gaussian_noise_std=0.01,
        )
        data = {
            "labels": torch.tensor([0, 1]),
            "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.2], [0.3, 0.3, 0.1, 0.1]]),
        }
        result = processor(data)
        assert "gold_bboxes" in result
        # Gold bboxes should preserve original values
        assert result["gold_bboxes"].shape == (2, 4)


class TestContentAwareProcessor:
    """Test ContentAwareProcessor class"""

    def test_initialization(self, sample_id2label, canvas_size):
        metadata = DataFrame(
            {
                "poster_path": ["train/0.png", "train/1.png"],
                "cls_elem": [1, 2],
                "box_elem": ["[0, 0, 100, 100]", "[50, 50, 150, 150]"],
            }
        )
        processor = ContentAwareProcessor(
            index2label=sample_id2label,
            canvas_width=canvas_size[0],
            canvas_height=canvas_size[1],
            metadata=metadata,
        )
        assert processor.metadata is metadata
        assert processor.max_element_numbers == 10

    def test_return_keys(self):
        assert ContentAwareProcessor.return_keys == [
            "idx",
            "labels",
            "bboxes",
            "gold_bboxes",
            "content_bboxes",
            "discrete_bboxes",
            "discrete_gold_bboxes",
            "discrete_content_bboxes",
        ]

    def test_normalize_bboxes(self, sample_id2label, canvas_size):
        metadata = DataFrame({"poster_path": [], "cls_elem": [], "box_elem": []})
        processor = ContentAwareProcessor(
            index2label=sample_id2label,
            canvas_width=canvas_size[0],
            canvas_height=canvas_size[1],
            metadata=metadata,
        )
        bboxes = torch.tensor([[513.0, 750.0, 256.5, 375.0]])
        normalized = processor._normalize_bboxes(bboxes)
        assert torch.allclose(normalized[0, 0], torch.tensor(1.0))
        assert torch.allclose(normalized[0, 1], torch.tensor(1.0))

    @patch("layoutprompter.preprocess.cv2.imread")
    def test_call_returns_none_for_empty_saliency(
        self, mock_imread, sample_id2label, canvas_size
    ):
        metadata = DataFrame({"poster_path": [], "cls_elem": [], "box_elem": []})
        processor = ContentAwareProcessor(
            index2label=sample_id2label,
            canvas_width=canvas_size[0],
            canvas_height=canvas_size[1],
            metadata=metadata,
        )
        # Mock empty saliency map (results in no bboxes)
        mock_imread.return_value = np.ones((100, 100, 3), dtype=np.uint8) * 50
        result = processor("dummy_path.png", "0", "train")
        assert result is None

    @patch("layoutprompter.preprocess.cv2.imread")
    def test_call_returns_none_for_no_labels(
        self, mock_imread, sample_id2label, canvas_size
    ):
        metadata = DataFrame(
            {
                "poster_path": ["train/0.png"],
                "cls_elem": [0],  # cls_elem <= 0
                "box_elem": ["[0, 0, 100, 100]"],
            }
        )
        processor = ContentAwareProcessor(
            index2label=sample_id2label,
            canvas_width=canvas_size[0],
            canvas_height=canvas_size[1],
            metadata=metadata,
        )
        # Mock saliency map that produces bboxes
        mock_imread.return_value = np.ones((100, 100, 3), dtype=np.uint8) * 200
        result = processor("dummy_path.png", "0", "train")
        assert result is None

    @patch("layoutprompter.preprocess.cv2.imread")
    def test_call_test_split_raises_without_training(
        self, mock_imread, sample_id2label, canvas_size
    ):
        metadata = DataFrame({"poster_path": [], "cls_elem": [], "box_elem": []})
        processor = ContentAwareProcessor(
            index2label=sample_id2label,
            canvas_width=canvas_size[0],
            canvas_height=canvas_size[1],
            metadata=metadata,
        )
        mock_imread.return_value = np.ones((100, 100, 3), dtype=np.uint8) * 200
        with pytest.raises(RuntimeError, match="Please process training data first"):
            processor("dummy_path.png", "0", "test")


class TestTextToLayoutProcessor:
    """Test TextToLayoutProcessor class"""

    def test_initialization(self, sample_id2label, canvas_size):
        processor = TextToLayoutProcessor(
            index2label=sample_id2label,
            canvas_width=canvas_size[0],
            canvas_height=canvas_size[1],
        )
        assert processor.index2label == sample_id2label
        assert processor.label2index == {v: k for k, v in sample_id2label.items()}

    def test_return_keys(self):
        assert TextToLayoutProcessor.return_keys == [
            "labels",
            "bboxes",
            "text",
            "embedding",
        ]

    def test_scale(self, sample_id2label, canvas_size):
        processor = TextToLayoutProcessor(
            index2label=sample_id2label,
            canvas_width=canvas_size[0],
            canvas_height=canvas_size[1],
        )
        elements = [
            {"type": "text", "position": [0, 0, 100, 100]},
            {"type": "image", "position": [100, 100, 200, 200]},
        ]
        scaled = processor._scale(720, elements)
        # Canvas width is 1440, original is 720, so ratio = 2
        assert scaled[0]["position"][0] == 0
        assert scaled[0]["position"][2] == 200
        assert scaled[1]["position"][0] == 200

    @patch("layoutprompter.preprocess.CLIPTextEncoder")
    def test_call(self, mock_encoder, sample_id2label, canvas_size):
        mock_encoder_instance = Mock()
        mock_encoder_instance.return_value = torch.randn(1, 512)
        mock_encoder.return_value = mock_encoder_instance

        processor = TextToLayoutProcessor(
            index2label=sample_id2label,
            canvas_width=canvas_size[0],
            canvas_height=canvas_size[1],
        )
        data = {
            "text": "Test layout description",
            "canvas_width": 720,
            "elements": [
                {"type": "text", "position": [10, 20, 100, 50]},
                {"type": "image", "position": [50, 100, 200, 150]},
            ],
        }
        result = processor(data)
        assert "text" in result
        assert "embedding" in result
        assert "labels" in result
        assert "discrete_gold_bboxes" in result
        assert result["labels"].shape[0] == 2


class TestCreateProcessor:
    """Test create_processor factory function"""

    def test_create_gent_processor(self):
        processor = create_processor("rico", "gent")
        assert isinstance(processor, GenTypeProcessor)

    def test_create_gents_processor(self):
        processor = create_processor("rico", "gents")
        assert isinstance(processor, GenTypeSizeProcessor)

    def test_create_genr_processor(self):
        processor = create_processor("rico", "genr")
        assert isinstance(processor, GenRelationProcessor)

    def test_create_completion_processor(self):
        processor = create_processor("rico", "completion")
        assert isinstance(processor, CompletionProcessor)

    def test_create_refinement_processor(self):
        processor = create_processor("rico", "refinement")
        assert isinstance(processor, RefinementProcessor)

    def test_create_content_processor(self):
        metadata = DataFrame({"poster_path": [], "cls_elem": [], "box_elem": []})
        processor = create_processor("posterlayout", "content", metadata=metadata)
        assert isinstance(processor, ContentAwareProcessor)

    def test_create_text_processor(self):
        processor = create_processor("webui", "text")
        assert isinstance(processor, TextToLayoutProcessor)

    def test_create_with_custom_args(self):
        processor = create_processor(
            "rico", "gent", sort_by_pos=True, shuffle_before_sort_by_label=False
        )
        assert isinstance(processor, GenTypeProcessor)
        assert processor.sort_by_pos is True
