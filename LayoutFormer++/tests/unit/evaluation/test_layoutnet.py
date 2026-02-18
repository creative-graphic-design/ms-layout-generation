import torch

from layoutformer_pp.evaluation.utils.layoutnet import LayoutNet, TransformerWithToken


def test_transformer_with_token_forward_shape():
    model = TransformerWithToken(d_model=8, nhead=2, dim_feedforward=16, num_layers=1)
    x = torch.randn(5, 2, 8)
    padding_mask = torch.tensor(
        [
            [False, False, True, True, True],
            [False, False, False, True, True],
        ]
    )

    out = model(x, padding_mask)

    assert out.shape == (6, 2, 8)
    assert model.token_mask.shape == (1, 1)


def test_layoutnet_forward_shapes():
    net = LayoutNet(num_label=4, max_bbox=5)
    bbox = torch.rand(2, 5, 4)
    label = torch.tensor(
        [
            [1, 2, 0, 0, 0],
            [3, 4, 1, 0, 0],
        ]
    )
    padding_mask = label == 0

    features = net.extract_features(bbox, label, padding_mask)
    logit_disc, logit_cls, bbox_pred = net(bbox, label, padding_mask)

    assert features.shape == (2, 256)
    assert logit_disc.shape == (2,)
    assert logit_cls.shape[1] == 5
    assert bbox_pred.shape[1] == 4
    assert logit_cls.shape[0] == (~padding_mask).sum().item()
