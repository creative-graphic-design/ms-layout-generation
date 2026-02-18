from __future__ import annotations

import torch

from eval_src.evaluation.utils.layoutnet import LayoutNet


def test_layoutnet_forward_shapes() -> None:
    model = LayoutNet(num_label=5, max_bbox=3)
    bbox = torch.rand(2, 3, 4)
    labels = torch.randint(0, 6, (2, 3))
    padding_mask = torch.tensor([[False, False, True], [False, True, True]])

    feats = model.extract_features(bbox, labels, padding_mask)
    assert feats.shape[0] == 2

    logit_disc, logit_cls, bbox_pred = model(bbox, labels, padding_mask)
    assert logit_disc.shape[0] == 2
    assert logit_cls.shape[1] == 6
    assert bbox_pred.shape[1] == 4
