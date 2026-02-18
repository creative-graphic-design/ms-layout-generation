from types import SimpleNamespace

import torch

from layoutformer_pp.tasks import task_utils
from layoutformer_pp.tasks.completion import T5CompletionDataset
from layoutformer_pp.tasks.refinement import T5RefinementDataset
from layoutformer_pp.tasks.task_config import create_seq_processor
from layoutformer_pp.tasks.task_utils import create_tokenizer
from layoutformer_pp.utils import utils
from layoutformer_pp.evaluation import metrics


class DummyModel:
    def __init__(self, tokenizer, output_strings):
        self.tokenizer = tokenizer
        self.output_strings = output_strings

    def __call__(self, in_ids, padding_mask, out_ids=None, **kwargs):
        if out_ids is not None:
            return {"loss": torch.ones(in_ids.size(0))}
        tokenized = self.tokenizer(self.output_strings, add_eos=True, add_bos=False)
        return {"output": tokenized["input_ids"]}


def _make_args():
    return SimpleNamespace(
        discrete_x_grid=16,
        discrete_y_grid=16,
        gaussian_noise_mean=0.0,
        gaussian_noise_std=0.0,
        train_bernoulli_beta=0.0,
        max_num_elements=5,
        dataset="rico",
    )


def test_evaluate_fn_topk_completion(rico_sample):
    args = _make_args()
    tokenizer = create_tokenizer(["completion"], "rico", discrete_grid=16)
    seq_processor = create_seq_processor("completion")(
        dataset="rico", tokenizer=tokenizer
    )
    dataset = T5CompletionDataset(args, tokenizer, seq_processor)

    processed = dataset.process(rico_sample)
    data = utils.collate_fn([processed, processed, processed])
    data["task_name"] = ["completion", "completion", "completion"]
    data["task_id"] = [1, 1, 1]

    short_str = (
        seq_processor.build_seq(
            processed["gold_labels"][:1], processed["gold_bboxes"][:1]
        )
        .lower()
        .strip()
    )
    long_str = f"{processed['out_str']} {processed['out_str']}"

    output_strings = ["bad output", short_str, long_str]
    model = DummyModel(tokenizer, output_strings)

    eval_fn = task_utils.EvaluateFn(max_num_elements=3, enable_task_measure=True)
    loss, prediction = eval_fn(
        model, data, seq_processor, tokenizer, device=torch.device("cpu")
    )

    assert loss.item() == 1.0
    assert prediction is not None
    assert prediction["pred_bboxes"].shape[-1] == 4


def test_evaluate_fn_greedy_refinement(rico_sample):
    args = _make_args()
    tokenizer = create_tokenizer(["refinement"], "rico", discrete_grid=16)
    seq_processor = create_seq_processor("refinement")(
        dataset="rico", tokenizer=tokenizer
    )
    dataset = T5RefinementDataset(args, tokenizer, seq_processor)

    processed = dataset.process(rico_sample)
    data = utils.collate_fn([processed])
    data["task_name"] = ["refinement"]
    data["task_id"] = [0]

    model = DummyModel(tokenizer, data["out_str"])
    eval_fn = task_utils.EvaluateFn(max_num_elements=5, enable_task_measure=True)

    loss, prediction = eval_fn(
        model, data, seq_processor, tokenizer, device=torch.device("cpu")
    )

    assert loss.item() == 1.0
    assert prediction is not None


def test_create_fid_model():
    args = SimpleNamespace(dataset="rico", max_num_elements=20)
    fid_model = task_utils.create_fid_model(args)
    assert isinstance(fid_model, metrics.LayoutFID)

    args.dataset = "publaynet"
    fid_model = task_utils.create_fid_model(args)
    assert isinstance(fid_model, metrics.LayoutFID)
