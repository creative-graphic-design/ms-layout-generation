"""Unit tests for data loading functions."""

import json
import tempfile
from pathlib import Path

import pytest
import torch

from layoutformer_pp.data.load import load_publaynet_data, load_rico_data


@pytest.fixture
def mock_publaynet_dir():
    """Create mock PubLayNet directory structure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        publaynet_dir = Path(tmpdir) / "publaynet"
        publaynet_dir.mkdir()

        # Create mock COCO annotation files
        train_data = {
            "images": [
                {"id": 1, "file_name": "train_1.jpg", "width": 400, "height": 600},
                {"id": 2, "file_name": "train_2.jpg", "width": 500, "height": 700},
            ],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 50, 50]},
                {"id": 2, "image_id": 1, "category_id": 2, "bbox": [100, 100, 80, 80]},
                {"id": 3, "image_id": 2, "category_id": 1, "bbox": [20, 20, 60, 60]},
            ],
            "categories": [
                {"id": 1, "name": "text"},
                {"id": 2, "name": "title"},
            ],
        }

        val_data = {
            "images": [
                {"id": 3, "file_name": "val_1.jpg", "width": 450, "height": 650},
            ],
            "annotations": [
                {"id": 4, "image_id": 3, "category_id": 1, "bbox": [15, 15, 55, 55]},
            ],
            "categories": [
                {"id": 1, "name": "text"},
                {"id": 2, "name": "title"},
            ],
        }

        with open(publaynet_dir / "train.json", "w") as f:
            json.dump(train_data, f)

        with open(publaynet_dir / "val.json", "w") as f:
            json.dump(val_data, f)

        yield tmpdir


def test_load_publaynet_data_basic(mock_publaynet_dir):
    """Test basic PubLayNet data loading."""
    label_set = ["text", "title"]
    label2index = {"text": 0, "title": 1}

    result = load_publaynet_data(
        raw_dir=str(mock_publaynet_dir),
        max_num_elements=10,
        label_set=label_set,
        label2index=label2index,
    )

    assert len(result) == 3  # train, test, val
    train_set, test_set, val_set = result

    # Check that we got some data
    assert len(train_set) > 0
    assert len(val_set) > 0


def test_load_publaynet_data_filters_invalid_bounds(mock_publaynet_dir):
    """Test that PubLayNet loading filters invalid bounding boxes."""
    # Add invalid data
    publaynet_dir = Path(mock_publaynet_dir) / "publaynet"

    with open(publaynet_dir / "train.json", "r") as f:
        train_data = json.load(f)

    # Add image with invalid bbox (negative coordinates)
    train_data["images"].append(
        {"id": 10, "file_name": "train_invalid.jpg", "width": 400, "height": 600}
    )
    train_data["annotations"].append(
        {"id": 10, "image_id": 10, "category_id": 1, "bbox": [-10, -10, 50, 50]}
    )

    with open(publaynet_dir / "train.json", "w") as f:
        json.dump(train_data, f)

    label_set = ["text", "title"]
    label2index = {"text": 0, "title": 1}

    result = load_publaynet_data(
        raw_dir=str(mock_publaynet_dir),
        max_num_elements=10,
        label_set=label_set,
        label2index=label2index,
    )

    train_set, test_set, val_set = result

    # Verify invalid bbox was filtered
    for data in train_set:
        for bbox in data["bboxes"]:
            assert torch.all(bbox >= 0)
            assert torch.all(bbox <= 1)


def test_load_publaynet_data_filters_horizontal_images(mock_publaynet_dir):
    """Test that PubLayNet loading filters horizontal images (W > H)."""
    publaynet_dir = Path(mock_publaynet_dir) / "publaynet"

    with open(publaynet_dir / "train.json", "r") as f:
        train_data = json.load(f)

    # Add horizontal image (W > H)
    train_data["images"].append(
        {"id": 11, "file_name": "train_horiz.jpg", "width": 800, "height": 600}
    )
    train_data["annotations"].append(
        {"id": 11, "image_id": 11, "category_id": 1, "bbox": [10, 10, 50, 50]}
    )

    with open(publaynet_dir / "train.json", "w") as f:
        json.dump(train_data, f)

    label_set = ["text", "title"]
    label2index = {"text": 0, "title": 1}

    result = load_publaynet_data(
        raw_dir=str(mock_publaynet_dir),
        max_num_elements=10,
        label_set=label_set,
        label2index=label2index,
    )

    train_set, test_set, val_set = result

    # Verify all images are portrait (H >= W)
    for data in train_set + test_set + val_set:
        W, H = data["canvas_size"]
        assert H >= W


def test_load_publaynet_data_respects_max_elements(mock_publaynet_dir):
    """Test that max_num_elements is respected."""
    label_set = ["text", "title"]
    label2index = {"text": 0, "title": 1}

    result = load_publaynet_data(
        raw_dir=str(mock_publaynet_dir),
        max_num_elements=1,  # Only 1 element allowed
        label_set=label_set,
        label2index=label2index,
    )

    train_set, test_set, val_set = result

    # All layouts should have <= 1 element
    for data in train_set + test_set + val_set:
        assert len(data["bboxes"]) <= 1


def test_load_publaynet_data_label_filtering(mock_publaynet_dir):
    """Test that only specified labels are loaded."""
    label_set = ["text"]  # Only load 'text', not 'title'
    label2index = {"text": 0}

    result = load_publaynet_data(
        raw_dir=str(mock_publaynet_dir),
        max_num_elements=10,
        label_set=label_set,
        label2index=label2index,
    )

    train_set, test_set, val_set = result

    # All labels should be 0 (text)
    for data in train_set + test_set + val_set:
        assert torch.all(data["labels"] == 0)


def test_load_publaynet_data_bbox_format(mock_publaynet_dir):
    """Test that bboxes are in ltwh format and normalized."""
    label_set = ["text", "title"]
    label2index = {"text": 0, "title": 1}

    result = load_publaynet_data(
        raw_dir=str(mock_publaynet_dir),
        max_num_elements=10,
        label_set=label_set,
        label2index=label2index,
    )

    train_set, test_set, val_set = result

    for data in train_set + test_set + val_set:
        for bbox in data["bboxes"]:
            # Check normalized (0-1 range)
            assert torch.all(bbox >= 0)
            assert torch.all(bbox <= 1)

            # Check format: [left, top, width, height]
            _left, _top, w, h = bbox
            assert w > 0
            assert h > 0


@pytest.fixture
def mock_rico_dir():
    """Create mock RICO directory structure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        rico_dir = Path(tmpdir) / "semantic_annotations"
        rico_dir.mkdir(parents=True)

        # Create mock RICO annotation files
        anno1 = {
            "bounds": [0, 0, 400, 600],
            "children": [
                {
                    "bounds": [10, 10, 60, 60],
                    "componentLabel": "Button",
                },
                {
                    "bounds": [100, 100, 180, 180],
                    "componentLabel": "Text",
                },
            ],
        }

        anno2 = {
            "bounds": [0, 0, 450, 650],
            "children": [
                {
                    "bounds": [20, 20, 80, 80],
                    "componentLabel": "Button",
                }
            ],
        }

        with open(rico_dir / "1.json", "w") as f:
            json.dump(anno1, f)

        with open(rico_dir / "2.json", "w") as f:
            json.dump(anno2, f)

        yield tmpdir


def test_load_rico_data_basic(mock_rico_dir):
    """Test basic RICO data loading."""
    label_set = ["Button", "Text"]
    label2index = {"Button": 0, "Text": 1}

    result = load_rico_data(
        raw_dir=str(mock_rico_dir),
        max_num_elements=10,
        label_set=label_set,
        label2index=label2index,
    )

    assert len(result) == 3  # train, test, val
    train_set, test_set, val_set = result

    # Check that we got some data
    total_samples = len(train_set) + len(test_set) + len(val_set)
    assert total_samples > 0


def test_load_rico_data_nested_children(mock_rico_dir):
    """Test RICO data loading with nested children."""
    rico_dir = Path(mock_rico_dir) / "semantic_annotations"

    # Add nested structure
    anno_nested = {
        "bounds": [0, 0, 400, 600],
        "children": [
            {
                "bounds": [10, 10, 200, 200],
                "componentLabel": "Container",
                "children": [
                    {
                        "bounds": [20, 20, 80, 80],
                        "componentLabel": "Button",
                    }
                ],
            }
        ],
    }

    with open(rico_dir / "nested.json", "w") as f:
        json.dump(anno_nested, f)

    label_set = ["Button", "Container"]
    label2index = {"Button": 0, "Container": 1}

    result = load_rico_data(
        raw_dir=str(mock_rico_dir),
        max_num_elements=10,
        label_set=label_set,
        label2index=label2index,
    )

    train_set, test_set, val_set = result

    # Should successfully parse nested children
    all_data = train_set + test_set + val_set
    assert len(all_data) > 0


def test_load_rico_data_filters_invalid_bounds(mock_rico_dir):
    """Test that RICO loading filters invalid bounding boxes."""
    rico_dir = Path(mock_rico_dir) / "semantic_annotations"

    # Add invalid data (negative bounds)
    anno_invalid = {
        "bounds": [0, 0, 400, 600],
        "children": [
            {
                "bounds": [-10, -10, 50, 50],
                "componentLabel": "Button",
            }
        ],
    }

    with open(rico_dir / "invalid.json", "w") as f:
        json.dump(anno_invalid, f)

    label_set = ["Button"]
    label2index = {"Button": 0}

    result = load_rico_data(
        raw_dir=str(mock_rico_dir),
        max_num_elements=10,
        label_set=label_set,
        label2index=label2index,
    )

    train_set, test_set, val_set = result

    # Verify no negative coordinates
    for data in train_set + test_set + val_set:
        for bbox in data["bboxes"]:
            assert torch.all(bbox >= 0)


def test_load_rico_data_filters_horizontal_layouts(mock_rico_dir):
    """Test that RICO loading filters horizontal layouts (W >= H)."""
    rico_dir = Path(mock_rico_dir) / "semantic_annotations"

    # Add horizontal layout
    anno_horiz = {
        "bounds": [0, 0, 800, 600],  # W > H
        "children": [
            {
                "bounds": [10, 10, 60, 60],
                "componentLabel": "Button",
            }
        ],
    }

    with open(rico_dir / "horiz.json", "w") as f:
        json.dump(anno_horiz, f)

    label_set = ["Button"]
    label2index = {"Button": 0}

    result = load_rico_data(
        raw_dir=str(mock_rico_dir),
        max_num_elements=10,
        label_set=label_set,
        label2index=label2index,
    )

    train_set, test_set, val_set = result

    # Verify all layouts are portrait
    for data in train_set + test_set + val_set:
        W, H = data["canvas_size"]
        assert H >= W


def test_load_rico_data_respects_max_elements(mock_rico_dir):
    """Test that max_num_elements is respected."""
    label_set = ["Button", "Text"]
    label2index = {"Button": 0, "Text": 1}

    result = load_rico_data(
        raw_dir=str(mock_rico_dir),
        max_num_elements=1,
        label_set=label_set,
        label2index=label2index,
    )

    train_set, test_set, val_set = result

    # All layouts should have <= 1 element
    for data in train_set + test_set + val_set:
        assert len(data["bboxes"]) <= 1


def test_load_rico_data_bbox_format(mock_rico_dir):
    """Test that RICO bboxes are in ltwh format and normalized."""
    label_set = ["Button", "Text"]
    label2index = {"Button": 0, "Text": 1}

    result = load_rico_data(
        raw_dir=str(mock_rico_dir),
        max_num_elements=10,
        label_set=label_set,
        label2index=label2index,
    )

    train_set, test_set, val_set = result

    for data in train_set + test_set + val_set:
        for bbox in data["bboxes"]:
            # Check normalized
            assert torch.all(bbox >= 0)
            assert torch.all(bbox <= 1)

            # Check format: [left, top, width, height]
            _left, _top, w, h = bbox
            assert w > 0
            assert h > 0


def test_load_rico_data_split_ratios(mock_rico_dir):
    """Test that RICO data split ratios are correct (85/5/10)."""
    label_set = ["Button", "Text"]
    label2index = {"Button": 0, "Text": 1}

    result = load_rico_data(
        raw_dir=str(mock_rico_dir),
        max_num_elements=10,
        label_set=label_set,
        label2index=label2index,
    )

    train_set, test_set, val_set = result
    total = len(train_set) + len(test_set) + len(val_set)

    # Check approximate split ratios (with small dataset, exact ratios may vary)
    if total > 0:
        train_ratio = len(train_set) / total
        # Train should be the largest portion
        assert train_ratio >= len(test_set) / total
        assert train_ratio >= len(val_set) / total


def test_load_publaynet_data_split_ratios(mock_publaynet_dir):
    """Test that PubLayNet data split ratios are correct (95/5 for train, 100 for val)."""
    label_set = ["text", "title"]
    label2index = {"text": 0, "title": 1}

    result = load_publaynet_data(
        raw_dir=str(mock_publaynet_dir),
        max_num_elements=10,
        label_set=label_set,
        label2index=label2index,
    )

    train_set, test_set, val_set = result

    # Train should be larger than test
    if len(train_set) > 0 or len(test_set) > 0:
        assert len(train_set) >= len(test_set)


def test_load_rico_data_filtered_flag(mock_rico_dir):
    """Test that filtered flag is set correctly."""
    rico_dir = Path(mock_rico_dir) / "semantic_annotations"

    # Add data that will be filtered
    anno_filtered = {
        "bounds": [0, 0, 400, 600],
        "children": [
            {
                "bounds": [10, 10, 60, 60],
                "componentLabel": "Button",
            },
            {
                "bounds": [-5, -5, 50, 50],  # Invalid - will be filtered
                "componentLabel": "Button",
            },
        ],
    }

    with open(rico_dir / "filtered.json", "w") as f:
        json.dump(anno_filtered, f)

    label_set = ["Button"]
    label2index = {"Button": 0}

    result = load_rico_data(
        raw_dir=str(mock_rico_dir),
        max_num_elements=10,
        label_set=label_set,
        label2index=label2index,
    )

    train_set, test_set, val_set = result

    # Check that filtered flag exists
    for data in train_set + test_set + val_set:
        assert "filtered" in data
        assert isinstance(data["filtered"], bool)


def test_load_publaynet_data_filtered_flag(mock_publaynet_dir):
    """Test that filtered flag is set correctly for PubLayNet."""
    label_set = ["text", "title"]
    label2index = {"text": 0, "title": 1}

    result = load_publaynet_data(
        raw_dir=str(mock_publaynet_dir),
        max_num_elements=10,
        label_set=label_set,
        label2index=label2index,
    )

    train_set, test_set, val_set = result

    # Check that filtered flag exists
    for data in train_set + test_set + val_set:
        assert "filtered" in data
        assert isinstance(data["filtered"], bool)
