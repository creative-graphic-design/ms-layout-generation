from types import SimpleNamespace

import torch

from layoutformer_pp.tasks.refinement import T5RefinementDataset, refinement_inference
from layoutformer_pp.tasks.task_config import create_seq_processor
from layoutformer_pp.tasks.task_utils import create_tokenizer, build_model
from layoutformer_pp.utils import utils


def _make_args():
    return SimpleNamespace(
        discrete_x_grid=16,
        discrete_y_grid=16,
        gaussian_noise_mean=0.0,
        gaussian_noise_std=0.0,
        train_bernoulli_beta=0.0,
        add_task_prompt=False,
        d_model=32,
        num_layers=2,
        nhead=4,
        dropout=0.1,
        share_embedding=False,
        num_pos_embed=128,
        add_task_embedding=False,
        add_task_prompt_token_in_model=False,
        num_task_prompt_token=1,
    )


def test_refinement_inference_end_to_end(rico_sample):
    args = _make_args()
    tokenizer = create_tokenizer(["refinement"], "rico", args.discrete_x_grid)
    seq_processor = create_seq_processor("refinement")(
        dataset="rico", tokenizer=tokenizer
    )
    dataset = T5RefinementDataset(args, tokenizer, seq_processor)

    processed = dataset.process(rico_sample)
    batch = utils.collate_fn([processed])

    model = build_model(args, tokenizer)
    metric, out = refinement_inference(
        model,
        batch,
        seq_processor=seq_processor,
        tokenizer=tokenizer,
        device=torch.device("cpu"),
    )

    assert metric["num_examples"] == 1
    assert out["pred_labels"].shape[0] == 1
    assert out["pred_bboxes"].shape[0] == 1
    assert out["mask"].shape[0] == 1
