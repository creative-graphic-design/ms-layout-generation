import copy

import torch

from layoutformer_pp.data.transforms import (
    AddGaussianNoise,
    DiscretizeBoundingBox,
    LabelDictSort,
    LexicographicSort,
)
from layoutformer_pp.data import RicoDataset


def test_discretize_continuize_roundtrip(rico_sample):
    bbox = rico_sample["bboxes"][:2]
    discretizer = DiscretizeBoundingBox(num_x_grid=32, num_y_grid=32)
    discrete = discretizer.discretize(bbox)
    continuous = discretizer.continuize(discrete)

    assert discrete.dtype == torch.long
    assert continuous.shape == bbox.shape
    assert torch.all((continuous >= 0) & (continuous <= 1))


def test_lexicographic_sort_orders_by_position(rico_sample):
    data = LexicographicSort()(copy.deepcopy(rico_sample))
    left, top, _, _ = data["bboxes"].t()
    pairs = list(zip(top.tolist(), left.tolist()))

    assert "gold_bboxes" in data
    assert pairs == sorted(pairs)


def test_label_dict_sort_orders_by_label_name(rico_sample):
    index2label = RicoDataset.index2label(RicoDataset.labels)
    data = LabelDictSort(index2label)(copy.deepcopy(rico_sample))
    label_names = [index2label[idx] for idx in data["labels"].tolist()]

    assert label_names == sorted(label_names)


def test_add_gaussian_noise_respects_bernoulli_gate(rico_sample):
    transform = AddGaussianNoise(mean=0.0, std=1.0, bernoulli_beta=0.0)
    data = transform(copy.deepcopy(rico_sample))

    assert torch.allclose(data["bboxes"], data["gold_bboxes"])
