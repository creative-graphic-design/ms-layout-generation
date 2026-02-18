from __future__ import annotations

import json
from pathlib import Path

from eval_src.data.load import load_publaynet_data, load_rico_data
from eval_src.data.base import PubLayNetDataset, RicoDataset


def _write_publaynet_json(path: Path) -> None:
    data = {
        "images": [{"id": 1, "width": 100, "height": 200, "file_name": "a.jpg"}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10]}
        ],
        "categories": [{"id": 1, "name": "text"}],
    }
    path.write_text(json.dumps(data), encoding="utf-8")


def test_load_publaynet_data(tmp_path: Path) -> None:
    raw_root = tmp_path / "publaynet"
    raw_root.mkdir(parents=True, exist_ok=True)
    _write_publaynet_json(raw_root / "train.json")
    _write_publaynet_json(raw_root / "val.json")

    label_set = PubLayNetDataset.labels
    label2index = PubLayNetDataset.label2index(label_set)
    splits = load_publaynet_data(
        str(tmp_path), max_num_elements=5, label_set=label_set, label2index=label2index
    )

    assert len(splits) == 3
    assert sum(len(split) for split in splits) > 0
    assert len(splits[2]) > 0


def test_load_rico_data(tmp_path: Path) -> None:
    raw_root = tmp_path / "semantic_annotations"
    raw_root.mkdir(parents=True, exist_ok=True)
    sample = {
        "bounds": [0, 0, 100, 200],
        "componentLabel": "Text",
        "children": [{"bounds": [10, 10, 20, 20], "componentLabel": "Text"}],
    }
    (raw_root / "sample.json").write_text(json.dumps(sample), encoding="utf-8")

    label_set = RicoDataset.labels
    label2index = RicoDataset.label2index(label_set)
    splits = load_rico_data(
        str(tmp_path), max_num_elements=5, label_set=label_set, label2index=label2index
    )

    assert len(splits) == 3
    assert len(splits[0]) >= 0
