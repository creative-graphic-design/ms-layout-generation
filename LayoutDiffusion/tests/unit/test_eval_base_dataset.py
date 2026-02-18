from __future__ import annotations

from pathlib import Path

from eval_src.data.base import RicoDataset, PubLayNetDataset


def test_label_mappings_and_colors(tmp_path: Path) -> None:
    labels = RicoDataset.labels
    label2index = RicoDataset.label2index(labels)
    index2label = RicoDataset.index2label(labels)

    assert label2index[labels[0]] == 1
    assert index2label[1] == labels[0]

    dataset = RicoDataset(root=str(tmp_path), split="train", max_num_elements=20)
    colors = dataset.colors
    assert len(colors) == len(labels) + 1


def test_publaynet_labels(tmp_path: Path) -> None:
    labels = PubLayNetDataset.labels
    label2index = PubLayNetDataset.label2index(labels)
    assert label2index[labels[0]] == 1

    dataset = PubLayNetDataset(root=str(tmp_path), split="val", max_num_elements=20)
    colors = dataset.colors
    assert len(colors) == len(labels) + 1
