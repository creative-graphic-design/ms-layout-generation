from __future__ import annotations

from pathlib import Path

import torch

from eval_src.utils import visualization


def test_convert_layout_to_image() -> None:
    boxes = [[0.0, 0.0, 0.5, 0.5], [0.5, 0.5, 1.0, 1.0]]
    labels = [1, 2]
    colors = [(0, 0, 0), (255, 0, 0), (0, 255, 0)]
    img = visualization.convert_layout_to_image(
        boxes, labels, colors, canvas_size=(20, 20)
    )
    assert img.size == (20, 20)


def test_save_image(tmp_path: Path) -> None:
    batch_boxes = torch.tensor([[[0.0, 0.0, 0.5, 0.5]]])
    batch_labels = torch.tensor([[1]])
    batch_mask = torch.tensor([[True]])
    colors = [(0, 0, 0), (255, 0, 0)]

    out_path = tmp_path / "img.png"
    visualization.save_image(
        batch_boxes,
        batch_labels,
        batch_mask,
        colors,
        str(out_path),
        canvas_size=(20, 20),
    )
    assert out_path.exists()
