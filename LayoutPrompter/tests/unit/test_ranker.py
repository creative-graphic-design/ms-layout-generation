import torch

from layoutprompter.ranker import Ranker
from layoutprompter.utils import (
    compute_alignment,
    compute_maximum_iou,
    compute_overlap,
    convert_ltwh_to_ltrb,
    write_pt,
)


def _make_prediction(overlap: bool):
    labels = torch.tensor([1, 1])
    if overlap:
        bboxes = torch.tensor([[0.1, 0.1, 0.4, 0.4], [0.2, 0.2, 0.4, 0.4]])
    else:
        bboxes = torch.tensor([[0.1, 0.1, 0.2, 0.2], [0.7, 0.7, 0.2, 0.2]])
    return labels, bboxes


def _compute_quality(predictions, val_labels=None, val_bboxes=None):
    metrics = []
    for pred_labels, pred_bboxes in predictions:
        metric = []
        _pred_labels = pred_labels.unsqueeze(0)
        _pred_bboxes = convert_ltwh_to_ltrb(pred_bboxes).unsqueeze(0)
        _pred_padding_mask = torch.ones_like(_pred_labels).bool()
        metric.append(compute_alignment(_pred_bboxes, _pred_padding_mask))
        metric.append(compute_overlap(_pred_bboxes, _pred_padding_mask))
        if val_labels is not None and val_bboxes is not None:
            metric.append(
                compute_maximum_iou(pred_labels, pred_bboxes, val_labels, val_bboxes)
            )
        metrics.append(metric)

    metrics = torch.tensor(metrics)
    min_vals, _ = torch.min(metrics, 0, keepdim=True)
    max_vals, _ = torch.max(metrics, 0, keepdim=True)
    scaled_metrics = (metrics - min_vals) / (max_vals - min_vals)
    if val_labels is not None and val_bboxes is not None:
        quality = scaled_metrics[:, 0] * 0.2 + scaled_metrics[:, 1] * 0.2 + (
            1 - scaled_metrics[:, 2]
        ) * 0.6
    else:
        quality = scaled_metrics[:, 0] * 0.2 + scaled_metrics[:, 1] * 0.2
    return quality.tolist()


def _prediction_index(pred, predictions):
    for idx, candidate in enumerate(predictions):
        if torch.equal(pred[0], candidate[0]) and torch.equal(pred[1], candidate[1]):
            return idx
    raise AssertionError("Prediction not found in original list")


def test_ranker_orders_by_quality_without_val():
    pred_low = _make_prediction(overlap=False)
    pred_high = _make_prediction(overlap=True)

    ranker = Ranker()
    predictions = [pred_high, pred_low]
    ranked = ranker(predictions)

    qualities = _compute_quality(predictions)
    ranked_qualities = [
        qualities[_prediction_index(prediction, predictions)] for prediction in ranked
    ]
    assert ranked_qualities == sorted(ranked_qualities)


def test_ranker_with_validation_data(tmp_path):
    val_path = tmp_path / "val.pt"
    val_data = [
        {
            "labels": torch.tensor([1, 1]),
            "bboxes": torch.tensor([[0.1, 0.1, 0.4, 0.4], [0.2, 0.2, 0.4, 0.4]]),
        }
    ]
    write_pt(val_path, val_data)

    ranker = Ranker(val_path=str(val_path))
    pred_match = _make_prediction(overlap=True)
    pred_mismatch = _make_prediction(overlap=False)

    predictions = [pred_match, pred_mismatch]
    ranked = ranker(predictions)
    val_labels = [item["labels"] for item in val_data]
    val_bboxes = [item["bboxes"] for item in val_data]
    qualities = _compute_quality(predictions, val_labels=val_labels, val_bboxes=val_bboxes)
    ranked_qualities = [
        qualities[_prediction_index(prediction, predictions)] for prediction in ranked
    ]
    assert ranked_qualities == sorted(ranked_qualities)
