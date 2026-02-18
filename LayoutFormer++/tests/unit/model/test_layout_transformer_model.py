import torch

from layoutformer_pp.model.layout_transformer.model import LayoutTransformer


class BacktrackConstraint:
    def __init__(self, vocab_size, eos_id):
        self.vocab_size = vocab_size
        self.eos_id = eos_id
        self._used = False

    def __call__(self, batch_id, idx, token_ids):
        if idx == 1 and not self._used:
            self._used = True
            return [self.eos_id], 0
        return list(range(self.vocab_size)), None


def _make_model(vocab_size=32, add_task_embedding=False, add_task_prompt=False):
    return LayoutTransformer(
        vocab_size=vocab_size,
        max_len=16,
        bos_token_id=1,
        pad_token_id=0,
        eos_token_id=2,
        d_model=16,
        nhead=2,
        num_layers=1,
        dropout=0.0,
        d_feedforward=32,
        share_embedding=False,
        add_task_embedding=add_task_embedding,
        num_task_embedding=3,
        add_task_prompt_token=add_task_prompt,
        num_task_prompt_token=2,
    )


def _make_inputs(batch=2, seq_len=4, vocab_size=32):
    input_ids = torch.randint(3, vocab_size, (batch, seq_len))
    padding_mask = torch.zeros(batch, seq_len, dtype=torch.bool)
    padding_mask[0, -1] = True
    return input_ids, padding_mask


def test_encode_with_task_prompt():
    model = _make_model(add_task_prompt=True)
    input_ids, padding_mask = _make_inputs()
    task_ids = torch.tensor([0, 1])

    enc_hs, enc_padding_mask = model.encode(input_ids, padding_mask, task_ids)

    assert enc_hs.shape[0] == input_ids.size(1) + 2
    assert enc_padding_mask.shape[1] == input_ids.size(1) + 2


def test_encode_with_task_embedding():
    model = _make_model(add_task_embedding=True)
    input_ids, padding_mask = _make_inputs()
    task_ids = torch.tensor([0, 2])

    enc_hs, enc_padding_mask = model.encode(input_ids, padding_mask, task_ids)

    assert enc_hs.shape[0] == input_ids.size(1)
    assert enc_padding_mask.shape == padding_mask.shape


def test_compute_loss_with_loss_weights():
    model = _make_model()
    input_ids, padding_mask = _make_inputs()
    labels = torch.randint(3, model.vocab_size, (2, input_ids.size(1)))
    loss_weights = torch.tensor([0.7, 1.3])

    outputs = model(input_ids, padding_mask, labels=labels, loss_weights=loss_weights)

    assert "loss" in outputs
    assert outputs["logits"].shape[0] == input_ids.size(0)


def test_generate_with_constraint():
    model = _make_model()
    input_ids, padding_mask = _make_inputs()

    def constraint_fn(batch_id, idx, token_ids):
        return [model.eos_token_id], None

    outputs = model.generate(
        input_ids,
        padding_mask,
        max_length=3,
        generation_constraint_fn=constraint_fn,
    )

    assert outputs["output"].shape[0] == input_ids.size(0)


def test_top_k_sample_path():
    model = _make_model()
    input_ids, padding_mask = _make_inputs()

    outputs = model(
        input_ids,
        padding_mask,
        max_length=3,
        do_sample=True,
        top_k=5,
        temperature=1.0,
    )

    assert outputs["output"].shape[0] == input_ids.size(0)


def test_decoding_space_restriction_backtracking():
    model = _make_model()
    input_ids, padding_mask = _make_inputs()
    constraint = BacktrackConstraint(model.vocab_size, model.eos_token_id)

    outputs = model.decoding_space_restriction(
        input_ids,
        padding_mask,
        max_length=4,
        generation_constraint_fn=constraint,
        top_k=5,
        temperature=1.0,
    )

    assert outputs["output"].shape[0] == input_ids.size(0)
