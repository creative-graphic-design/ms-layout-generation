from layoutformer_pp.model.layout_transformer.tokenizer import (
    LayoutTransformerTokenizer,
)
from layoutformer_pp.tasks.refinement import T5LayoutSequence
from layoutformer_pp.tasks.completion import T5CompletionLayoutSequence
from layoutformer_pp.tasks.task_utils import EvaluateFn


def _make_tokenizer():
    tokens = ["label1", "0", "1", "2", "3", "4"]
    return LayoutTransformerTokenizer(tokens)


def test_evaluate_fn_parse_seq_handles_signature_variants():
    tokenizer = _make_tokenizer()
    index2label = {1: "label1"}
    label2index = {"label1": 1}

    seq_single = T5LayoutSequence(tokenizer, index2label, label2index)
    seq_double = T5CompletionLayoutSequence(tokenizer, index2label, label2index)

    eval_fn = EvaluateFn(max_num_elements=5)
    seq = "label1 1 2 3 4"

    labels_single, bbox_single = eval_fn._parse_seq(seq_single, seq)
    labels_double, bbox_double = eval_fn._parse_seq(seq_double, seq)

    assert labels_single == [1]
    assert bbox_single == [[1, 2, 3, 4]]
    assert labels_double == [1]
    assert bbox_double == [[1, 2, 3, 4]]
