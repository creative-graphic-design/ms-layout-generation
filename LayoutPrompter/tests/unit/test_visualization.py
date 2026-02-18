import os
from pathlib import Path

import torch
from PIL import Image

from layoutprompter.utils import RAW_DATA_PATH, CANVAS_SIZE
from layoutprompter.visualization import (
    ContentAwareVisualizer,
    Visualizer,
    create_image_grid,
)


def test_visualizer_draw_layout_and_colors():
    visualizer = Visualizer(dataset="rico", times=2)
    labels = torch.tensor([1, 2])
    bboxes = torch.tensor([[0.1, 0.1, 0.2, 0.2], [0.6, 0.6, 0.2, 0.2]])

    img = visualizer.draw_layout(labels, bboxes)
    assert isinstance(img, Image.Image)

    width, height = CANVAS_SIZE["rico"]
    assert img.size == (width * 2, height * 2)

    colors_first = visualizer.colors
    colors_second = visualizer.colors
    assert colors_first is colors_second

    # Pixel inside the first box should not be pure white
    sample_pixel = img.getpixel((int(width * 0.2), int(height * 0.2)))
    assert sample_pixel != (255, 255, 255)


def test_create_image_grid_dimensions():
    img = Image.new("RGB", (10, 10), color=(255, 0, 0))
    grid = create_image_grid([img, img, img, img], rows=2, cols=2, border_size=2)
    assert grid.size == (26, 26)


def test_content_aware_visualizer_with_temp_canvas():
    raw_path = Path(RAW_DATA_PATH("posterlayout"))
    canvas_dir = raw_path / "test" / "image_canvas"
    canvas_dir.mkdir(parents=True, exist_ok=True)
    canvas_path = canvas_dir / "0.png"

    created = False
    if not canvas_path.exists():
        img = Image.new("RGB", (513, 750), color=(255, 255, 255))
        img.save(canvas_path)
        created = True

    visualizer = ContentAwareVisualizer(times=1)
    labels = torch.tensor([1, 2])
    bboxes = torch.tensor([[0.1, 0.1, 0.2, 0.2], [0.5, 0.5, 0.2, 0.2]])
    images = visualizer([(labels, bboxes)], test_idx=0)

    assert len(images) == 1
    assert isinstance(images[0], Image.Image)

    if created:
        canvas_path.unlink()
        if not any(canvas_dir.iterdir()):
            canvas_dir.rmdir()
            if not any((raw_path / "test").iterdir()):
                (raw_path / "test").rmdir()
