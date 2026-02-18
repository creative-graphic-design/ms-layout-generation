import torch

from layoutformer_pp.utils import visualization


def test_convert_layout_to_image_returns_canvas():
    boxes = [[0.0, 0.0, 1.0, 1.0]]
    labels = [0]
    colors = [(255, 0, 0)]

    img = visualization.convert_layout_to_image(
        boxes, labels, colors, canvas_size=(10, 20)
    )

    assert img.size == (20, 10)
    assert img.getpixel((1, 1)) != (255, 255, 255)


def test_save_image_writes_file(tmp_path):
    batch_boxes = torch.tensor(
        [
            [[0.0, 0.0, 1.0, 1.0], [0.1, 0.1, 0.2, 0.2]],
            [[0.0, 0.0, 0.5, 0.5], [0.5, 0.5, 1.0, 1.0]],
        ]
    )
    batch_labels = torch.tensor([[0, 1], [1, 0]])
    batch_mask = torch.tensor([[True, False], [True, True]])
    colors = [(255, 0, 0), (0, 255, 0)]
    out_path = tmp_path / "layout.png"

    visualization.save_image(
        batch_boxes,
        batch_labels,
        batch_mask,
        colors,
        str(out_path),
        canvas_size=(10, 10),
        nrow=1,
    )

    assert out_path.exists()
    assert out_path.stat().st_size > 0
