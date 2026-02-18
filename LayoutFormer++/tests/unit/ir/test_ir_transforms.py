import torch

from layoutformer_pp.data import transforms


def test_discretize_continuize_round_trip():
    discretizer = transforms.DiscretizeBoundingBox(num_x_grid=4, num_y_grid=4)
    bbox = torch.tensor([[0.0, 0.5, 1.0, 0.25]])

    discrete = discretizer.discretize(bbox)
    assert discrete.tolist() == [[0, 1, 3, 0]]

    continuous = discretizer.continuize(discrete)
    expected = torch.tensor([[0.0, 1.0 / 3.0, 1.0, 0.0]])
    torch.testing.assert_close(continuous, expected)


def test_coordinate_transform_xywh_preserves_ltwh():
    data = {
        "bboxes": torch.tensor([[0.0, 0.0, 1.0, 2.0]]),
        "canvas_size": [1.0, 1.0],
    }
    transform = transforms.CoordinateTransform("xywh")

    out = transform({**data})

    assert "bboxes_ltwh" in out
    assert "gold_bboxes_ltwh" in out
    torch.testing.assert_close(out["bboxes_ltwh"], data["bboxes"])

    expected = torch.tensor([[0.5, 1.0, 1.0, 2.0]])
    torch.testing.assert_close(out["bboxes"], expected)


def test_add_gaussian_noise_zero_std_is_noop():
    data = {
        "bboxes": torch.tensor([[0.1, 0.2, 0.3, 0.4]]),
        "canvas_size": [1.0, 1.0],
    }
    transform = transforms.AddGaussianNoise(
        mean=0.0, std=0.0, normalized=True, bernoulli_beta=1.0
    )

    out = transform({**data})

    torch.testing.assert_close(out["bboxes"], data["bboxes"])
