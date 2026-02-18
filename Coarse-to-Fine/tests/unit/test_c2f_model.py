"""Unit tests for c2f_model/model.py classes"""

import torch
import pytest
from types import SimpleNamespace

from coarse_to_fine.c2f_model.model import (
    LayoutEmbedding,
    Encoder,
    VAE,
    GroupDecoder,
    ElementDecoder,
    C2FLayoutTransformer,
)
from coarse_to_fine.c2f_utils import padding


@pytest.fixture
def cfg():
    """Create a minimal config object for testing"""
    return SimpleNamespace(
        num_labels=5,
        discrete_x_grid=32,
        discrete_y_grid=32,
        d_model=128,
        n_heads=4,
        dim_feedforward=256,
        dropout=0.1,
        n_layers=2,
        n_layers_decoder=2,
        d_z=128,
        eval_batch_size=4,
        max_num_elements=10,
    )


@pytest.fixture
def layout_embd(cfg):
    """Create a LayoutEmbedding instance"""
    return LayoutEmbedding(cfg)


class TestLayoutEmbedding:
    """Tests for LayoutEmbedding class"""

    def test_initialization(self, cfg):
        """Test LayoutEmbedding initialization"""
        embd = LayoutEmbedding(cfg)
        assert embd.cfg == cfg
        assert embd.label_embed is not None
        assert embd.bbox_embed is not None
        assert embd.proj_cat is not None
        assert embd.group_label_embed is not None

    def test_get_label_embedding(self, layout_embd, cfg):
        """Test label embedding"""
        labels = torch.zeros(2, 3, dtype=torch.long)
        emb = layout_embd.get_label_embedding(labels)
        assert emb.shape == (2, 3, 128)

    def test_get_box_embedding(self, layout_embd, cfg):
        """Test box embedding"""
        boxes = torch.zeros(2, 3, 4, dtype=torch.long)
        emb = layout_embd.get_box_embedding(boxes)
        # Shape: (S, N, 4*128) -> (S, N, 512)
        assert emb.shape == (2, 3, 512)

    def test_get_group_label_embedding(self, layout_embd, cfg):
        """Test group label embedding"""
        group_labels = torch.zeros(2, 3, cfg.num_labels + 2)
        emb = layout_embd.get_group_label_embedding(group_labels)
        assert emb.shape == (2, 3, 128)

    def test_forward(self, layout_embd, cfg):
        """Test forward pass"""
        labels = torch.zeros(2, 3, dtype=torch.long)
        boxes = torch.zeros(2, 3, 4, dtype=torch.long)
        output = layout_embd(labels, boxes)
        # proj_cat reduces 128 + 512 = 640 to d_model
        assert output.shape == (2, 3, cfg.d_model)


class TestEncoder:
    """Tests for Encoder class"""

    def test_initialization(self, cfg, layout_embd):
        """Test Encoder initialization"""
        encoder = Encoder(cfg, layout_embd)
        assert encoder.embedding == layout_embd
        assert encoder.encoder is not None

    def test_forward(self, cfg, layout_embd):
        """Test forward pass"""
        encoder = Encoder(cfg, layout_embd)
        labels = torch.zeros(2, 3, dtype=torch.long)
        bboxes = torch.zeros(2, 3, 4, dtype=torch.long)
        masks = torch.ones(3, 2, dtype=torch.bool)  # (batch_size, seq_len)

        memory = encoder(labels, bboxes, masks)
        # Output is pooled: (1, batch_size, d_model)
        assert memory.shape == (1, 3, cfg.d_model)


class TestVAE:
    """Tests for VAE class"""

    def test_initialization(self, cfg):
        """Test VAE initialization"""
        vae = VAE(cfg)
        assert vae.cfg == cfg
        assert vae.enc_mu_fcn is not None
        assert vae.enc_sigma_fcn is not None
        assert vae.z_fcn is not None

    def test_forward(self, cfg):
        """Test forward pass"""
        vae = VAE(cfg)
        memory = torch.randn(1, 3, cfg.d_model)
        z, mu, logvar = vae(memory)

        assert z.shape == (1, 3, cfg.d_z)
        assert mu.shape == (1, 3, cfg.d_z)
        assert logvar.shape == (1, 3, cfg.d_z)

    def test_inference_with_none(self, cfg):
        """Test inference with z=None"""
        vae = VAE(cfg)
        device = torch.device("cpu")
        z = vae.inference(None, device)

        assert z.shape == (1, cfg.eval_batch_size, cfg.d_z)

    def test_inference_with_z(self, cfg):
        """Test inference with provided z"""
        vae = VAE(cfg)
        device = torch.device("cpu")
        # _make_seq_first expects 3D tensor (batch, seq, features)
        input_z = torch.randn(2, 3, cfg.d_z)
        z = vae.inference(input_z, device)

        # After permute: (seq, batch, features)
        assert z.shape == (3, 2, cfg.d_z)


class TestGroupDecoder:
    """Tests for GroupDecoder class"""

    def test_initialization(self, cfg, layout_embd):
        """Test GroupDecoder initialization"""
        decoder = GroupDecoder(cfg, layout_embd)
        assert decoder.cfg == cfg
        assert decoder.layout_embd == layout_embd
        assert decoder.decoder is not None
        assert decoder.label_fcn is not None
        assert decoder.box_fcn is not None

    def test_forward_shape(self, cfg, layout_embd):
        """Test forward pass shape"""
        decoder = GroupDecoder(cfg, layout_embd)

        # Create minimal inputs
        G = 3  # num groups
        N = 2  # batch size
        label = torch.zeros(G, N, cfg.num_labels + 2)
        box = torch.zeros(G, N, 4, dtype=torch.long)
        z = torch.randn(1, N, cfg.d_model)
        mask = torch.ones(N, G, dtype=torch.bool)

        out, rec_box, rec_label = decoder(label, box, z, mask)

        # out shape: (G-2, N, E) - removes eos
        assert out.shape[0] == G - 2
        assert out.shape[1] == N
        assert out.shape[2] == cfg.d_model

        # rec_box shape: (G, N, 4*d_box)
        d_box = max(cfg.discrete_x_grid, cfg.discrete_y_grid)
        assert rec_box.shape == (G, N, 4 * d_box)

        # rec_label shape: (G, N, num_labels+2)
        assert rec_label.shape == (G, N, cfg.num_labels + 2)

    def test_inference_shape(self, cfg, layout_embd):
        decoder = GroupDecoder(cfg, layout_embd)
        device = torch.device("cpu")
        z = torch.randn(1, 2, cfg.d_model)
        out, rec_box, rec_label = decoder.inference(z, device, max_group_num=4)
        d_box = max(cfg.discrete_x_grid, cfg.discrete_y_grid)
        assert out.shape[1] == 2
        assert rec_box.shape == (4, 2, 4 * d_box)
        assert rec_label.shape == (4, 2, cfg.num_labels + 2)


class TestElementDecoder:
    """Tests for ElementDecoder class"""

    def test_initialization(self, cfg, layout_embd):
        """Test ElementDecoder initialization"""
        decoder = ElementDecoder(cfg, layout_embd)
        assert decoder.cfg == cfg
        assert decoder.layout_embd == layout_embd
        assert decoder.decoder is not None
        assert decoder.label_fcn is not None
        assert decoder.box_fcn is not None

    def test_forward_shape(self, cfg, layout_embd):
        """Test forward pass shape"""
        decoder = ElementDecoder(cfg, layout_embd)

        # Create minimal inputs
        S = 4  # sequence length
        G = 2  # num groups
        N = 2  # batch size
        label = torch.zeros(S, G, N, 1, dtype=torch.long)
        box = torch.zeros(S, G, N, 4, dtype=torch.long)
        memory = torch.randn(G, N, cfg.d_model)
        z = torch.randn(1, N, cfg.d_model)
        mask = torch.ones(N, G, S, dtype=torch.bool)

        rec_box, rec_label = decoder(label, box, memory, z, mask)

        # rec_box shape: (S, G, N, 4*d_box)
        d_box = max(cfg.discrete_x_grid, cfg.discrete_y_grid)
        assert rec_box.shape == (S, G, N, 4 * d_box)

        # rec_label shape: (S, G, N, num_labels+3)
        assert rec_label.shape == (S, G, N, cfg.num_labels + 3)

    def test_inference_shape(self, cfg, layout_embd):
        decoder = ElementDecoder(cfg, layout_embd)
        device = torch.device("cpu")
        memory = torch.randn(2, 3, cfg.d_model)
        z = torch.randn(1, 3, cfg.d_model)
        rec_box, rec_label = decoder.inference(memory, z, max_num_elements=4, device=device)
        d_box = max(cfg.discrete_x_grid, cfg.discrete_y_grid)
        assert rec_box.shape == (4, 2, 3, 4 * d_box)
        assert rec_label.shape == (4, 2, 3, cfg.num_labels + 3)


class TestC2FLayoutTransformer:
    def _make_raw_data(self, cfg):
        return {
            "labels": [torch.tensor([1, 2, 3])],
            "bboxes": [
                torch.tensor(
                    [
                        [1, 1, 2, 2],
                        [3, 3, 2, 2],
                        [5, 5, 2, 2],
                    ],
                    dtype=torch.long,
                )
            ],
            "label_in_one_group": [
                torch.zeros(3, cfg.num_labels + 2)
            ],
            "group_bounding_box": [
                torch.tensor(
                    [
                        [0.0, 0.0, 1.0, 1.0],
                        [0.0, 0.0, 1.0, 1.0],
                        [0.0, 0.0, 1.0, 1.0],
                    ],
                    dtype=torch.long,
                )
            ],
            "grouped_label": [[torch.tensor([1, 2, 3])]],
            "grouped_box": [
                [
                    torch.tensor(
                        [
                            [1, 1, 2, 2],
                            [3, 3, 2, 2],
                            [5, 5, 2, 2],
                        ],
                        dtype=torch.long,
                    )
                ]
            ],
        }

    def test_initialization(self, cfg):
        model = C2FLayoutTransformer(cfg)
        assert model.cfg == cfg
        assert isinstance(model.layout_embd, LayoutEmbedding)
        assert isinstance(model.encoder, Encoder)
        assert isinstance(model.vae, VAE)
        assert isinstance(model.group_decoder, GroupDecoder)
        assert isinstance(model.ele_decoder, ElementDecoder)

    def test_forward_feed_gt(self, cfg):
        device = torch.device("cpu")
        data = padding(self._make_raw_data(cfg), device)
        model = C2FLayoutTransformer(cfg)

        ori, rec, kl_info = model(data, device, feed_gt=True)

        assert "bboxes" in ori
        assert "grouped_bboxes" in rec
        assert "mu" in kl_info and "logvar" in kl_info

    def test_forward_inference_path(self, cfg):
        device = torch.device("cpu")
        data = padding(self._make_raw_data(cfg), device)
        model = C2FLayoutTransformer(cfg)

        ori, rec, kl_info = model(data, device, feed_gt=False)

        assert rec["grouped_bboxes"].shape[0] == ori["bboxes"].shape[0]
        assert "z" in kl_info

    def test_inference_outputs(self, cfg):
        device = torch.device("cpu")
        model = C2FLayoutTransformer(cfg)

        gen = model.inference(device)

        assert "group_bounding_box" in gen
        assert "label_in_one_group" in gen
        assert "grouped_bboxes" in gen
        assert "grouped_labels" in gen
