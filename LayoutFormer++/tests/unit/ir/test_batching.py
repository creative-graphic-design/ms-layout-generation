import torch

from layoutformer_pp.utils import utils


def test_to_dense_batch_pads_and_masks():
    batch = [torch.tensor([1, 2]), torch.tensor([3])]

    dense, mask = utils.to_dense_batch(batch)

    assert dense.tolist() == [[1, 2], [3, 0]]
    assert mask.tolist() == [[True, True], [True, False]]
