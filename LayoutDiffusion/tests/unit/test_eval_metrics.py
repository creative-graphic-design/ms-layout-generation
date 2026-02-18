from __future__ import annotations

import numpy as np
import pytest
import torch

from eval_src.evaluation import metrics as metrics_module
from eval_src.evaluation.metrics import (
    LayoutFID,
    calculate_bbox_accuracy,
    calculate_label_accuracy,
    check_labels,
    compute_iou,
    compute_alignment,
    compute_maximum_iou,
    compute_overlap,
    compute_overlap_ignore_bg,
    compute_self_sim,
    average,
)
from eval_src.evaluation.utils.layoutnet import LayoutNet


def test_overlap_and_alignment_scores() -> None:
    bboxes = torch.tensor([[[0.0, 0.0, 0.5, 0.5], [0.5, 0.5, 1.0, 1.0]]])
    labels = torch.tensor([[1, 4]])
    mask = torch.tensor([[True, True]])

    overlap = compute_overlap(bboxes, mask)
    overlap_ignore = compute_overlap_ignore_bg(bboxes, labels, mask)
    align = compute_alignment(bboxes, mask)

    assert overlap >= 0.0
    assert overlap_ignore >= 0.0
    assert align >= 0.0


def test_maximum_iou_and_self_sim() -> None:
    layout_a = (np.array([[0.0, 0.0, 1.0, 1.0]]), np.array([1]))
    layout_b = (np.array([[0.0, 0.0, 1.0, 1.0]]), np.array([1]))
    score = compute_maximum_iou([layout_a], [layout_b], n_jobs=1)
    assert score == 1.0

    self_sim = compute_self_sim([layout_a], [layout_b], n_jobs=1)
    assert self_sim >= 0.0


def test_label_and_bbox_accuracy() -> None:
    gold_labels = torch.tensor([[1, 2]])
    pred_labels = torch.tensor([[1, 3]])
    mask = torch.tensor([[True, True]])

    correct, total = calculate_label_accuracy(
        gold_labels, pred_labels, mask, element_wise=False
    )
    assert total == 1
    assert correct == 0

    bboxes_gold = torch.tensor([[[0.0, 0.0, 1.0, 1.0]]])
    bboxes_pred = torch.tensor([[[0.0, 0.0, 1.0, 1.0]]])
    bbox_correct, bbox_total = calculate_bbox_accuracy(
        bboxes_gold, bboxes_pred, torch.tensor([[True]])
    )
    assert bbox_correct == bbox_total


def test_layout_fid_collects_and_scores(tmp_path) -> None:
    num_labels = 2
    max_elements = 2
    layoutnet = LayoutNet(num_label=num_labels, max_bbox=max_elements)
    state_dict = {f"module.{k}": v for k, v in layoutnet.state_dict().items()}
    ckpt_path = tmp_path / "layoutnet.pt"
    torch.save(state_dict, ckpt_path)

    fid = LayoutFID(max_elements, num_labels, str(ckpt_path), device="cpu")
    bboxes = torch.rand(2, max_elements, 4)
    labels = torch.randint(0, num_labels + 1, (2, max_elements))
    padding_mask = torch.zeros(2, max_elements, dtype=torch.bool)

    fid.collect_features(bboxes, labels, padding_mask, real=False)
    fid.collect_features(bboxes, labels, padding_mask, real=True)
    score_1 = fid.compute_score()
    assert np.isfinite(score_1)

    fid.collect_features(bboxes, labels, padding_mask, real=True)
    fid.collect_features(bboxes, labels, padding_mask, real=False)
    score_2 = fid.compute_score()
    assert np.isfinite(score_2)


def test_compute_iou_with_numpy_torch_and_invalid() -> None:
    box_np = np.array([[0.0, 0.0, 1.0, 1.0]])
    iou_np = compute_iou(box_np, box_np)
    assert iou_np.shape == (1,)
    assert np.isclose(iou_np[0], 1.0)

    box_t = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
    iou_t = compute_iou(box_t, box_t)
    assert torch.allclose(iou_t, torch.ones(1))

    with pytest.raises(NotImplementedError):
        compute_iou([[0, 0, 1, 1]], [[0, 0, 1, 1]])


def test_internal_iou_helpers_and_cond_map() -> None:
    layout = (np.array([[0.0, 0.0, 1.0, 1.0]]), np.array([1]))
    layout_b = (np.array([[0.0, 0.0, 1.0, 1.0]]), np.array([1]))

    max_iou_for_layout = getattr(metrics_module, "__compute_maximum_iou_for_layout")
    score = max_iou_for_layout(layout, layout_b)
    assert np.isclose(score, 1.0)

    max_iou = getattr(metrics_module, "__compute_maximum_iou")
    scores = max_iou(([layout], [layout_b]))
    assert scores.shape == (1,)

    mean_iou_for_layout = getattr(metrics_module, "__compute_mean_iou_for_layout")
    mean_score = mean_iou_for_layout(layout, layout_b)
    assert np.isfinite(mean_score)

    mean_iou = getattr(metrics_module, "__compute_mean_iou")
    mean_pair = mean_iou(([layout, layout_b], [layout, layout_b]))
    assert np.isfinite(mean_pair)

    cond_map = getattr(metrics_module, "__get_cond2layouts")([layout, layout_b])
    assert list(cond_map.values())[0]


def test_compute_self_sim_multiple_layouts() -> None:
    layout = (np.array([[0.0, 0.0, 1.0, 1.0]]), np.array([1]))
    score = compute_self_sim([layout, layout], [layout, layout], n_jobs=1)
    assert score >= 0.0


def test_check_labels_and_accuracy_element_wise() -> None:
    assert check_labels(None, [1]) is False
    assert check_labels([1], None) is False
    assert check_labels([1], [1, 2]) is False
    assert check_labels([1, 2], [1, 2]) is True
    assert check_labels([1, 2], [2, 1]) is False

    gold = torch.tensor([[1, 2]])
    pred = torch.tensor([[1, 3]])
    mask = torch.tensor([[True, True]])
    correct, total = calculate_label_accuracy(gold, pred, mask, element_wise=True)
    assert total == 2
    assert correct == 1

    bboxes_gold = torch.tensor([[[0.0, 0.0, 1.0, 1.0], [0.1, 0.1, 0.2, 0.2]]])
    bboxes_pred = torch.tensor([[[0.0, 0.0, 1.0, 1.0], [0.1, 0.1, 0.0, 0.2]]])
    bbox_correct, bbox_total = calculate_bbox_accuracy(
        bboxes_gold, bboxes_pred, torch.tensor([[True, True]]), element_wise=True
    )
    assert bbox_total == 2
    assert bbox_correct == 1


def test_average() -> None:
    assert average([1, 2, 3]) == 2
