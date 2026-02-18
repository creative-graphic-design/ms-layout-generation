import math

import numpy as np
import torch

from layoutprompter.utils import (
    clean_text,
    compute_alignment,
    compute_maximum_iou,
    compute_overlap,
    convert_ltwh_to_ltrb,
    detect_loc_relation,
    detect_size_relation,
    labels_bboxes_similarity,
    labels_similarity,
    bboxes_similarity,
)


def test_clean_text_basic_and_remove_summary():
    text = "Hello,#Summary# world\nNew,line.#"
    cleaned = clean_text(text)
    assert cleaned == "Hello, Summary world New, line."

    cleaned_removed = clean_text(text, remove_summary=True)
    assert cleaned_removed == "Hello, world New, line."


def test_convert_ltwh_to_ltrb_single_and_batch():
    single = torch.tensor([1.0, 2.0, 3.0, 4.0])
    l, t, r, b = convert_ltwh_to_ltrb(single)
    assert torch.isclose(l, torch.tensor(1.0))
    assert torch.isclose(t, torch.tensor(2.0))
    assert torch.isclose(r, torch.tensor(4.0))
    assert torch.isclose(b, torch.tensor(6.0))

    batch = torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.1, 0.2]])
    ltrb = convert_ltwh_to_ltrb(batch)
    expected = torch.tensor([[0.1, 0.2, 0.4, 0.6], [0.5, 0.6, 0.6, 0.8]])
    assert torch.allclose(ltrb, expected)


def test_detect_size_relation_variants():
    b1 = torch.tensor([0.0, 0.0, 1.0, 1.0])
    b2_smaller = torch.tensor([0.0, 0.0, 0.5, 0.5])
    b2_equal = torch.tensor([0.0, 0.0, 1.0, 1.05])
    b2_larger = torch.tensor([0.0, 0.0, 1.5, 1.0])

    assert detect_size_relation(b1, b2_smaller) == "smaller"
    assert detect_size_relation(b1, b2_equal) == "equal"
    assert detect_size_relation(b1, b2_larger) == "larger"


def test_detect_loc_relation_canvas_and_elements():
    b1 = torch.tensor([0.4, 0.4, 0.2, 0.2])
    b2_top = torch.tensor([0.4, 0.1, 0.2, 0.1])
    b2_bottom = torch.tensor([0.4, 0.7, 0.2, 0.1])
    b2_left = torch.tensor([0.1, 0.4, 0.2, 0.2])
    b2_right = torch.tensor([0.7, 0.4, 0.2, 0.2])
    b2_center = torch.tensor([0.45, 0.45, 0.1, 0.1])

    assert detect_loc_relation(b1, b2_top, canvas=False) == "top"
    assert detect_loc_relation(b1, b2_bottom, canvas=False) == "bottom"
    assert detect_loc_relation(b1, b2_left, canvas=False) == "left"
    assert detect_loc_relation(b1, b2_right, canvas=False) == "right"
    assert detect_loc_relation(b1, b2_center, canvas=False) == "center"

    b2_canvas_top = torch.tensor([0.0, 0.0, 0.1, 0.1])
    b2_canvas_center = torch.tensor([0.0, 0.4, 0.1, 0.2])
    b2_canvas_bottom = torch.tensor([0.0, 0.8, 0.1, 0.2])
    assert detect_loc_relation(b1, b2_canvas_top, canvas=True) == "top"
    assert detect_loc_relation(b1, b2_canvas_center, canvas=True) == "center"
    assert detect_loc_relation(b1, b2_canvas_bottom, canvas=True) == "bottom"


def test_similarity_functions_labels_and_bboxes():
    labels_1 = torch.tensor([1, 1, 2])
    labels_2 = torch.tensor([1, 2, 2])
    sim = labels_similarity(labels_1, labels_2)
    assert math.isclose(sim, 4 / 6)

    bboxes_1 = torch.tensor([[0.1, 0.1], [0.5, 0.5]])
    bboxes_2 = torch.tensor([[0.1, 0.1], [0.5, 0.5]])
    label_same = torch.tensor([1, 2])
    bbox_sim = bboxes_similarity(label_same, bboxes_1, label_same, bboxes_2)
    assert math.isclose(bbox_sim, 1.0, rel_tol=1e-6)

    combined = labels_bboxes_similarity(
        label_same, bboxes_1, label_same, bboxes_2, labels_weight=0.2, bboxes_weight=0.8
    )
    assert math.isclose(combined, 1.0, rel_tol=1e-6)


def test_compute_maximum_iou_prefers_best_match():
    labels = torch.tensor([1, 2])
    bboxes = torch.tensor([[0.1, 0.1], [0.5, 0.5]])
    labels_list = [labels, torch.tensor([2, 1])]
    bboxes_list = [bboxes, torch.tensor([[0.2, 0.2], [0.6, 0.6]])]

    score = compute_maximum_iou(labels, bboxes, labels_list, bboxes_list)
    assert math.isclose(score, 1.0, rel_tol=1e-6)


def test_alignment_and_overlap_scores_are_reasonable():
    bboxes = torch.tensor(
        [
            [[0.1, 0.1, 0.2, 0.2], [0.7, 0.7, 0.2, 0.2]],
        ]
    )
    mask = torch.tensor([[True, True]])
    no_overlap = compute_overlap(bboxes, mask)

    overlap_bboxes = torch.tensor(
        [
            [[0.1, 0.1, 0.4, 0.4], [0.2, 0.2, 0.4, 0.4]],
        ]
    )
    overlap = compute_overlap(overlap_bboxes, mask)

    assert 0.0 <= no_overlap <= 1.0
    assert 0.0 <= overlap <= 1.0
    assert overlap >= no_overlap

    alignment_score = compute_alignment(overlap_bboxes, mask)
    assert alignment_score >= 0.0
    assert np.isfinite(alignment_score)
