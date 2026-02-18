import torch
import pytest

from layoutprompter.transforms import CLIPTextEncoder


@pytest.mark.unit
@pytest.mark.slow
def test_clip_text_encoder_real_model():
    encoder = CLIPTextEncoder()
    embedding = encoder("A simple layout description")
    assert isinstance(embedding, torch.Tensor)
    assert embedding.ndim == 2
    norms = embedding.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-3)
