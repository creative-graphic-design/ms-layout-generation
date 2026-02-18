import numpy as np
import pytest

from layoutformer_pp.trainer.utils import CheckpointMeasurement


def _make_layouts(bboxes_list):
    layouts = []
    for bboxes in bboxes_list:
        labels = np.ones(len(bboxes), dtype=np.int64)
        layouts.append((np.array(bboxes, dtype=np.float32), labels))
    return layouts


def test_checkpoint_measurement_miou_and_update():
    gold_layouts = _make_layouts([[[0, 0, 1, 1]]])
    pred_layouts = _make_layouts([[[0, 0, 1, 1]]])
    measure = CheckpointMeasurement(
        max_num_elements=2, measurement=CheckpointMeasurement.MIOU
    )

    value = measure.compute(gold_layouts, pred_layouts)
    assert 0.0 <= value <= 1.0

    assert measure.update(value) is True
    assert measure.update(value) is False


def test_checkpoint_measurement_alignment_overlap_and_reset():
    pred_layouts = _make_layouts(
        [
            [[0, 0, 1, 1], [2, 2, 1, 1]],
            [[0, 0, 2, 2]],
        ]
    )
    gold_layouts = pred_layouts

    alignment = CheckpointMeasurement(
        max_num_elements=3, measurement=CheckpointMeasurement.ALIGNMENT
    )
    overlap = CheckpointMeasurement(
        max_num_elements=3, measurement=CheckpointMeasurement.OVERLAP
    )

    alignment_value = alignment.compute(gold_layouts, pred_layouts)
    overlap_value = overlap.compute(gold_layouts, pred_layouts)

    assert isinstance(alignment_value, float)
    assert isinstance(overlap_value, float)

    assert alignment.update(alignment_value) is True
    alignment.reset()
    assert alignment.best_value == float("inf")

    assert overlap.update(overlap_value) is True
    overlap.reset()
    assert overlap.best_value == float("inf")


def test_checkpoint_measurement_invalid_type():
    with pytest.raises(NotImplementedError):
        CheckpointMeasurement(max_num_elements=2, measurement="unknown")
