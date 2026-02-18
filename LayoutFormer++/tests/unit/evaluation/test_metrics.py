import numpy as np
import pytest
import torch

from layoutformer_pp.evaluation import metrics
from layoutformer_pp.evaluation.utils.layoutnet import LayoutNet


def test_compute_iou_numpy_and_torch():
    box_1 = np.array([[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0]])
    box_2 = np.array([[0.0, 0.0, 1.0, 1.0], [1.0, 1.0, 2.0, 2.0]])
    iou_np = metrics.compute_iou(box_1, box_2)
    assert np.allclose(iou_np, [1.0, 0.0])

    iou_torch = metrics.compute_iou(torch.tensor(box_1), torch.tensor(box_2))
    expected = torch.tensor([1.0, 0.0], dtype=iou_torch.dtype)
    assert torch.allclose(iou_torch, expected)


def test_compute_iou_invalid_type():
    with pytest.raises(NotImplementedError):
        metrics.compute_iou([[0, 0, 1, 1]], [[0, 0, 1, 1]])


def test_compute_maximum_iou_helpers():
    bboxes = np.array([[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 2.0, 2.0]])
    labels = np.array([1, 2])
    layout = (bboxes, labels)

    score = metrics.__compute_maximum_iou_for_layout(layout, layout)
    assert score == pytest.approx(1.0)

    scores = metrics.__compute_maximum_iou(([layout], [layout]))
    assert scores.shape == (1,)
    assert scores[0] == pytest.approx(1.0)


def test_compute_maximum_iou_mp():
    bboxes = np.array([[0.0, 0.0, 1.0, 1.0]])
    labels = np.array([1])
    layout = (bboxes, labels)

    score = metrics.compute_maximum_iou([layout], [layout], n_jobs=1)
    assert score == pytest.approx(1.0)


def test_overlap_alignment_and_ignore_bg():
    bbox = torch.tensor([[[0.0, 0.0, 1.0, 1.0], [0.5, 0.5, 1.5, 1.5]]])
    mask = torch.tensor([[True, True]])

    overlap = metrics.compute_overlap(bbox, mask)
    assert overlap.shape == (1,)
    assert overlap.item() > 0.0

    alignment = metrics.compute_alignment(bbox, mask)
    assert alignment.shape == (1,)
    assert torch.isfinite(alignment).all()

    labels = torch.tensor([[4, 1]])
    ignore_overlap = metrics.compute_overlap_ignore_bg(bbox, labels, mask)
    assert ignore_overlap.item() == pytest.approx(0.0)


def test_label_and_bbox_accuracy_and_average():
    gold = torch.tensor([[1, 2], [1, 2]])
    pred = torch.tensor([[1, 2], [2, 2]])
    mask = torch.tensor([[True, True], [True, True]])

    num_correct, total = metrics.calculate_label_accuracy(
        gold, pred, mask, element_wise=False
    )
    assert num_correct.item() == 1
    assert total == 2

    num_correct, total = metrics.calculate_label_accuracy(
        gold, pred, mask, element_wise=True
    )
    assert num_correct.item() == 3
    assert total == 4

    gold_bbox = torch.tensor([[[0, 0, 1, 1], [0, 0, 2, 2]]])
    pred_bbox = torch.tensor([[[0, 0, 1, 1], [0, 1, 2, 2]]])
    mask_bbox = torch.tensor([[True, True]])

    num_correct, total = metrics.calculate_bbox_accuracy(
        gold_bbox, pred_bbox, mask_bbox, element_wise=False
    )
    assert total.item() == 8
    assert num_correct.item() == 7

    num_correct, total = metrics.calculate_bbox_accuracy(
        gold_bbox, pred_bbox, mask_bbox, element_wise=True
    )
    assert total.item() == 2
    assert num_correct.item() == 1

    assert metrics.check_labels(["a", "b"], ["a", "b"]) is True
    assert metrics.check_labels(["a"], ["b"]) is False
    assert metrics.check_labels(None, ["a"]) is False

    assert metrics.average([1.0, 3.0]) == 2.0


def test_layout_fid_collect_and_compute(tmp_path):
    net_path = tmp_path / "fid_tmp.pth.tar"
    model = LayoutNet(num_label=2, max_bbox=3)
    torch.save(model.state_dict(), net_path)

    fid = metrics.LayoutFID(
        max_num_elements=3, num_labels=2, net_path=str(net_path), device="cpu"
    )

    for _ in range(2):
        bbox = torch.rand(1, 3, 4)
        labels = torch.tensor([[1, 2, 0]])
        padding_mask = torch.tensor([[False, False, True]])
        fid.collect_features(bbox, labels, padding_mask, real=True)
        fid.collect_features(bbox, labels, padding_mask, real=False)

    score = fid.compute_score()
    assert np.isfinite(score)
