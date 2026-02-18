from __future__ import annotations

import pickle
from pathlib import Path

import torch

from eval_src.tools.draw_from_results import draw_images_from_results


def test_draw_images_from_results(tmp_path: Path) -> None:
    result_path = tmp_path / "results.pt"
    payload = [
        {
            "pred": [
                torch.tensor([[0.1, 0.1, 0.5, 0.5]]).numpy(),
                torch.tensor([1]).numpy(),
            ]
        }
    ]
    with result_path.open("wb") as f:
        pickle.dump(payload, f)

    save_path = tmp_path / "out"
    draw_images_from_results(
        dataset="rico",
        dataset_dir=str(tmp_path),
        max_num_elements=20,
        path=str(result_path),
        save_path=str(save_path),
        num_images=1,
    )

    assert any(save_path.glob("*.png"))
