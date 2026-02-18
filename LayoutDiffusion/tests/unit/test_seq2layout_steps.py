from __future__ import annotations

import os
import pickle
from pathlib import Path

import torch

from util_scripts import seq2layout_steps


def test_continuize_with_ltrb_hint() -> None:
    bbox = [[0, 0, 127, 127]]
    out = seq2layout_steps.continuize(bbox, "sample_ltrb.json")
    assert out.shape == (1, 4)
    assert torch.all(out >= 0)
    assert torch.all(out <= 1)


def test_generate_layouts_creates_pt(tmp_path: Path) -> None:
    json_path = tmp_path / "sample_elem1.json"
    json_path.write_text('["START Text 0 0 10 10 END"]\n', encoding="utf-8")

    output_paths = seq2layout_steps.generate_layouts(str(json_path), max_elems=1)
    assert len(output_paths) == 1

    out_path = Path(output_paths[0])
    assert out_path.exists()

    with out_path.open("rb") as f:
        payload = pickle.load(f)

    assert isinstance(payload, list)
    assert payload
    assert "pred" in payload[0]


def test_decapulate_and_continuize_ltwh() -> None:
    bbox = torch.tensor([[[0, 0, 10, 10], [10, 10, 20, 20]]])
    x1, y1, x2, y2 = seq2layout_steps.decapulate(bbox)
    assert x1.shape == (1, 2)
    assert y2.shape == (1, 2)

    out = seq2layout_steps.continuize([[0, 0, 10, 10]], "sample_ltwh.json")
    assert out.shape == (1, 4)
    assert torch.all(out >= 0)
    assert torch.all(out <= 1)


def test_clean_extract_split_and_build_layouts(tmp_path: Path) -> None:
    lines = [
        "START Text 0 0 10 10 | Image 0 0 5 5 \n",
        "| START Icon 0 0 1 1 \n",
    ]
    cleaned = seq2layout_steps._clean_lines(lines)
    assert cleaned[0][-1] != ""

    layouts = seq2layout_steps._extract_layouts(cleaned)
    assert layouts

    layouts_sep = seq2layout_steps._split_layouts(
        [
            [
                "Text",
                "0",
                "0",
                "10",
                "10",
                "|",
                "Image",
                "0",
                "0",
                "5",
                "5",
                "|",
                "Icon",
                "0",
                "0",
                "1",
                "1",
            ]
        ]
    )
    assert layouts_sep and len(layouts_sep[0]) >= 2

    layouts_final = seq2layout_steps._build_layout_outputs(
        [
            [
                [],
                ["Text", "x", "y", "z", "w"],
                ["Unknown", "1", "0", "0", "10", "10"],
                ["Text", "1", "0", "0", "10", "10"],
                ["Icon", "1", "0", "0", "5", "5"],
            ]
        ],
        "sample_ltrb.json",
    )
    assert layouts_final
    assert "pred" in layouts_final[0]


def test_main_runs_and_draws(tmp_path: Path) -> None:
    json_path = tmp_path / "sample_elem1.json"
    json_path.write_text('["START Text 0 0 10 10 END"]\n', encoding="utf-8")

    cwd = Path.cwd()
    try:
        os.chdir(Path(seq2layout_steps.__file__).resolve().parents[1])
        result = seq2layout_steps.main([str(json_path), "1"])
    finally:
        os.chdir(cwd)
    assert result == 0

    out_dir = tmp_path / "sample_out1"
    assert any(out_dir.glob("*.png"))
