import torch
import pytest

from layoutprompter.selection import (
    CompletionExemplarSelection,
    ContentAwareExemplarSelection,
    ExemplarSelection,
    GenRelationExemplarSelection,
    GenTypeExemplarSelection,
    GenTypeSizeExemplarSelection,
    RefinementExemplarSelection,
    TextToLayoutExemplarSelection,
    create_selector,
)


def _nonzero_discrete_bboxes(count):
    return torch.tensor([[1, 1, 1, 1]] * count)


@pytest.mark.unit
def test_is_filter_flags_zero_width_or_height():
    selector = ExemplarSelection(train_data=[], candidate_size=0, num_prompt=1)
    data_zero = {"discrete_gold_bboxes": torch.tensor([[0, 0, 1, 0]])}
    data_ok = {"discrete_gold_bboxes": torch.tensor([[0, 0, 1, 1]])}

    assert selector._is_filter(data_zero) is True
    assert selector._is_filter(data_ok) is False


@pytest.mark.unit
def test_retrieve_exemplars_skips_filtered_items():
    train_data = [
        {
            "labels": torch.tensor([1]),
            "discrete_gold_bboxes": torch.tensor([[0, 0, 0, 0]]),
        },
        {
            "labels": torch.tensor([2]),
            "discrete_gold_bboxes": torch.tensor([[1, 1, 1, 1]]),
        },
    ]
    selector = ExemplarSelection(
        train_data=train_data,
        candidate_size=0,
        num_prompt=1,
        shuffle=False,
    )

    exemplars = selector._retrieve_exemplars([[0, 1.0], [1, 0.5]])

    assert len(exemplars) == 1
    assert exemplars[0] is train_data[1]


@pytest.mark.unit
def test_gen_type_selection_prefers_label_overlap():
    train_data = [
        {
            "labels": torch.tensor([1, 1]),
            "discrete_gold_bboxes": _nonzero_discrete_bboxes(2),
        },
        {
            "labels": torch.tensor([1, 2]),
            "discrete_gold_bboxes": _nonzero_discrete_bboxes(2),
        },
        {
            "labels": torch.tensor([2, 2]),
            "discrete_gold_bboxes": _nonzero_discrete_bboxes(2),
        },
    ]
    selector = GenTypeExemplarSelection(
        train_data=train_data,
        candidate_size=0,
        num_prompt=1,
        shuffle=False,
    )
    test_data = {"labels": torch.tensor([1, 2])}

    exemplars = selector(test_data)

    assert exemplars[0] is train_data[1]


@pytest.mark.unit
def test_gen_type_size_selection_prefers_bbox_similarity():
    train_data = [
        {
            "labels": torch.tensor([1, 2]),
            "bboxes": torch.tensor([[0.0, 0.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]),
            "discrete_gold_bboxes": _nonzero_discrete_bboxes(2),
        },
        {
            "labels": torch.tensor([1, 2]),
            "bboxes": torch.tensor([[5.0, 5.0, 2.0, 2.0], [6.0, 6.0, 2.0, 2.0]]),
            "discrete_gold_bboxes": _nonzero_discrete_bboxes(2),
        },
    ]
    selector = GenTypeSizeExemplarSelection(
        train_data=train_data,
        candidate_size=0,
        num_prompt=1,
        shuffle=False,
    )
    test_data = {
        "labels": torch.tensor([1, 2]),
        "bboxes": torch.tensor([[0.0, 0.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]),
    }

    exemplars = selector(test_data)

    assert exemplars[0] is train_data[0]


@pytest.mark.unit
def test_gen_relation_selection_matches_labels():
    train_data = [
        {
            "labels": torch.tensor([1]),
            "discrete_gold_bboxes": _nonzero_discrete_bboxes(1),
        },
        {
            "labels": torch.tensor([2]),
            "discrete_gold_bboxes": _nonzero_discrete_bboxes(1),
        },
    ]
    selector = GenRelationExemplarSelection(
        train_data=train_data,
        candidate_size=0,
        num_prompt=1,
        shuffle=False,
    )
    test_data = {"labels": torch.tensor([2])}

    exemplars = selector(test_data)

    assert exemplars[0] is train_data[1]


@pytest.mark.unit
def test_completion_selection_uses_first_element_only():
    train_data = [
        {
            "labels": torch.tensor([1, 2]),
            "bboxes": torch.tensor([[0.0, 0.0, 1.0, 1.0], [9.0, 9.0, 1.0, 1.0]]),
            "discrete_gold_bboxes": _nonzero_discrete_bboxes(2),
        },
        {
            "labels": torch.tensor([1, 2]),
            "bboxes": torch.tensor([[5.0, 5.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0]]),
            "discrete_gold_bboxes": _nonzero_discrete_bboxes(2),
        },
    ]
    selector = CompletionExemplarSelection(
        train_data=train_data,
        candidate_size=0,
        num_prompt=1,
        shuffle=False,
    )
    test_data = {
        "labels": torch.tensor([1, 2]),
        "bboxes": torch.tensor([[0.0, 0.0, 1.0, 1.0], [8.0, 8.0, 1.0, 1.0]]),
    }

    exemplars = selector(test_data)

    assert exemplars[0] is train_data[0]


@pytest.mark.unit
def test_refinement_selection_prefers_full_bbox_similarity():
    train_data = [
        {
            "labels": torch.tensor([1]),
            "bboxes": torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
            "discrete_gold_bboxes": _nonzero_discrete_bboxes(1),
        },
        {
            "labels": torch.tensor([1]),
            "bboxes": torch.tensor([[4.0, 4.0, 1.0, 1.0]]),
            "discrete_gold_bboxes": _nonzero_discrete_bboxes(1),
        },
    ]
    selector = RefinementExemplarSelection(
        train_data=train_data,
        candidate_size=0,
        num_prompt=1,
        shuffle=False,
    )
    test_data = {
        "labels": torch.tensor([1]),
        "bboxes": torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
    }

    exemplars = selector(test_data)

    assert exemplars[0] is train_data[0]


@pytest.mark.unit
def test_content_aware_selection_uses_iou():
    train_data = [
        {
            "discrete_content_bboxes": torch.tensor([[0, 0, 10, 10]]),
            "discrete_gold_bboxes": _nonzero_discrete_bboxes(1),
        },
        {
            "discrete_content_bboxes": torch.tensor([[50, 50, 10, 10]]),
            "discrete_gold_bboxes": _nonzero_discrete_bboxes(1),
        },
    ]
    selector = ContentAwareExemplarSelection(
        train_data=train_data,
        candidate_size=0,
        num_prompt=1,
        shuffle=False,
    )
    test_data = {"discrete_content_bboxes": torch.tensor([[0, 0, 10, 10]])}

    exemplars = selector(test_data)

    assert exemplars[0] is train_data[0]


@pytest.mark.unit
def test_text_to_layout_selection_uses_embedding_similarity():
    train_data = [
        {
            "embedding": torch.tensor([[1.0, 0.0]]),
            "discrete_gold_bboxes": _nonzero_discrete_bboxes(1),
        },
        {
            "embedding": torch.tensor([[0.0, 1.0]]),
            "discrete_gold_bboxes": _nonzero_discrete_bboxes(1),
        },
    ]
    selector = TextToLayoutExemplarSelection(
        train_data=train_data,
        candidate_size=0,
        num_prompt=1,
        shuffle=False,
    )
    test_data = {"embedding": torch.tensor([[1.0, 0.0]])}

    exemplars = selector(test_data)

    assert exemplars[0] is train_data[0]


@pytest.mark.unit
def test_create_selector_returns_expected_type():
    selector = create_selector(
        task="gent",
        train_data=[
            {
                "labels": torch.tensor([1]),
                "discrete_gold_bboxes": _nonzero_discrete_bboxes(1),
            }
        ],
        candidate_size=0,
        num_prompt=1,
    )

    assert isinstance(selector, GenTypeExemplarSelection)
