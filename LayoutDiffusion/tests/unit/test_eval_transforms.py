from __future__ import annotations

import torch

from eval_src.data import transforms


def test_shuffle_elements_adds_gold_bboxes_and_preserves_values() -> None:
    data = {
        "bboxes": torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]),
        "labels": torch.tensor([2, 1]),
    }
    transform = transforms.ShuffleElements()
    out = transform(data)

    assert "gold_bboxes" in out
    assert out["bboxes"].shape == (2, 4)
    assert sorted(out["labels"].tolist()) == [1, 2]
    assert sorted(out["gold_bboxes"].tolist()) == sorted(data["bboxes"].tolist())


def test_label_dict_sort_orders_by_label_name() -> None:
    index2label = {1: "A", 2: "B"}
    data = {
        "bboxes": torch.tensor([[0.0, 0.0, 0.1, 0.1], [0.2, 0.2, 0.3, 0.3]]),
        "labels": torch.tensor([2, 1]),
    }
    sorter = transforms.LabelDictSort(index2label=index2label)
    out = sorter(data)

    assert out["labels"].tolist() == [1, 2]
    assert out["bboxes"].shape == (2, 4)
    assert out["gold_bboxes"].shape == (2, 4)


def test_lexicographic_sort_uses_top_then_left() -> None:
    data = {
        "bboxes": torch.tensor([[0.5, 0.5, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1]]),
        "labels": torch.tensor([1, 2]),
    }
    sorter = transforms.LexicographicSort()
    out = sorter(data)

    assert out["labels"].tolist() == [2, 1]
    assert out["bboxes"].shape == (2, 4)
    assert out["gold_bboxes"].shape == (2, 4)


def test_add_gaussian_noise_with_zero_std_no_change() -> None:
    data = {
        "bboxes": torch.tensor([[0.2, 0.2, 0.4, 0.4]]),
        "labels": torch.tensor([1]),
    }
    noise = transforms.AddGaussianNoise(
        mean=0.0, std=0.0, normalized=True, bernoulli_beta=1.0
    )
    out = noise(data)

    assert torch.allclose(out["bboxes"], data["bboxes"])
    assert torch.all(out["bboxes"] >= 0)
    assert torch.all(out["bboxes"] <= 1)


def test_coordinate_transform_ltrb() -> None:
    data = {"bboxes": torch.tensor([[0.1, 0.2, 0.3, 0.4]])}
    original = data["bboxes"].clone()
    transform = transforms.CoordinateTransform("ltrb")
    out = transform(data)

    expected = transforms.convert_ltwh_to_ltrb(original)
    assert torch.allclose(out["bboxes"], expected)
    assert torch.allclose(out["gold_bboxes"], expected)


def test_discretize_bounding_box_roundtrip() -> None:
    data = {"bboxes": torch.tensor([[0.1, 0.2, 0.3, 0.4]])}
    discretizer = transforms.DiscretizeBoundingBox(128, 128)
    out = discretizer(data)

    assert out["discrete_bboxes"].dtype == torch.long
    cont = discretizer.continuize(out["discrete_bboxes"])
    assert cont.shape == (1, 4)


def test_convert_and_decapulate_helpers() -> None:
    bbox = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
    ltrb = transforms.convert_ltwh_to_ltrb(bbox)
    xywh = transforms.convert_ltwh_to_xywh(bbox)
    back = transforms.convert_xywh_to_ltrb(xywh)

    x1, y1, x2, y2 = transforms.decapulate(ltrb)
    assert torch.allclose(x1, ltrb[:, 0])
    assert torch.allclose(y1, ltrb[:, 1])
    assert torch.allclose(x2, ltrb[:, 2])
    assert torch.allclose(y2, ltrb[:, 3])
    assert torch.allclose(ltrb, back)
