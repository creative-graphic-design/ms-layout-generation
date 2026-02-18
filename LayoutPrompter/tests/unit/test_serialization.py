"""Comprehensive unit tests for serialization.py"""

import pytest
import torch

from layoutprompter.serialization import (
    HTML_PREFIX,
    HTML_SUFFIX,
    HTML_TEMPLATE,
    HTML_TEMPLATE_WITH_INDEX,
    PREAMBLE,
    CompletionSerializer,
    ContentAwareSerializer,
    GenRelationSerializer,
    GenTypeSerializer,
    GenTypeSizeSerializer,
    RefinementSerializer,
    Serializer,
    TextToLayoutSerializer,
    build_prompt,
    create_serializer,
)


@pytest.mark.unit
class TestSerializer:
    """Test base Serializer class"""

    def test_initialization(self):
        serializer = Serializer(
            input_format="seq",
            output_format="seq",
            index2label={0: "text", 1: "title"},
            canvas_width=100,
            canvas_height=200,
            add_index_token=True,
            add_sep_token=True,
            add_unk_token=False,
        )
        assert serializer.input_format == "seq"
        assert serializer.output_format == "seq"
        assert serializer.canvas_width == 100
        assert serializer.canvas_height == 200
        assert serializer.add_index_token is True
        assert serializer.add_sep_token is True
        assert serializer.sep_token == "|"

    def test_initialization_with_custom_tokens(self):
        serializer = Serializer(
            input_format="seq",
            output_format="seq",
            index2label={0: "text"},
            canvas_width=100,
            canvas_height=200,
            sep_token=";",
            unk_token="<UNK>",
        )
        assert serializer.sep_token == ";"
        assert serializer.unk_token == "<UNK>"

    def test_build_output_unsupported_format_raises_none(self):
        serializer = Serializer(
            input_format="seq",
            output_format="invalid",
            index2label={0: "text"},
            canvas_width=100,
            canvas_height=200,
        )
        data = {
            "labels": torch.tensor([0]),
            "discrete_gold_bboxes": torch.tensor([[10, 20, 30, 40]]),
        }
        # Should return None for unsupported format
        result = serializer.build_output(data)
        assert result is None

    def test_build_input_seq_raises_not_implemented(self):
        serializer = Serializer(
            input_format="seq",
            output_format="seq",
            index2label={0: "text"},
            canvas_width=100,
            canvas_height=200,
        )
        with pytest.raises(NotImplementedError):
            serializer.build_input({})

    def test_build_input_html_raises_not_implemented(self):
        serializer = Serializer(
            input_format="html",
            output_format="seq",
            index2label={0: "text"},
            canvas_width=100,
            canvas_height=200,
        )
        with pytest.raises(NotImplementedError):
            serializer.build_input({})

    def test_build_input_unsupported_format(self):
        serializer = Serializer(
            input_format="invalid",
            output_format="seq",
            index2label={0: "text"},
            canvas_width=100,
            canvas_height=200,
        )
        with pytest.raises(ValueError, match="Unsupported input format"):
            serializer.build_input({})


@pytest.mark.unit
class TestSerializerSeqOutput:
    """Test Serializer seq output format"""

    def test_seq_output_with_index_and_sep(self):
        serializer = Serializer(
            input_format="seq",
            output_format="seq",
            index2label={0: "text", 1: "title"},
            canvas_width=100,
            canvas_height=200,
            add_index_token=True,
            add_sep_token=True,
            add_unk_token=False,
        )
        data = {
            "labels": torch.tensor([0, 1]),
            "discrete_gold_bboxes": torch.tensor([[10, 20, 30, 40], [1, 2, 3, 4]]),
        }
        output = serializer.build_output(data)
        assert output == "text 0 10 20 30 40 | title 1 1 2 3 4"

    def test_seq_output_without_index(self):
        serializer = Serializer(
            input_format="seq",
            output_format="seq",
            index2label={0: "text", 1: "title"},
            canvas_width=100,
            canvas_height=200,
            add_index_token=False,
            add_sep_token=True,
        )
        data = {
            "labels": torch.tensor([0, 1]),
            "discrete_gold_bboxes": torch.tensor([[10, 20, 30, 40], [1, 2, 3, 4]]),
        }
        output = serializer.build_output(data)
        assert output == "text 10 20 30 40 | title 1 2 3 4"

    def test_seq_output_without_sep(self):
        serializer = Serializer(
            input_format="seq",
            output_format="seq",
            index2label={0: "text", 1: "title"},
            canvas_width=100,
            canvas_height=200,
            add_index_token=True,
            add_sep_token=False,
        )
        data = {
            "labels": torch.tensor([0, 1]),
            "discrete_gold_bboxes": torch.tensor([[10, 20, 30, 40], [1, 2, 3, 4]]),
        }
        output = serializer.build_output(data)
        assert output == "text 0 10 20 30 40 title 1 1 2 3 4"

    def test_seq_output_with_custom_label_key(self):
        serializer = Serializer(
            input_format="seq",
            output_format="seq",
            index2label={0: "text"},
            canvas_width=100,
            canvas_height=200,
            add_index_token=False,
            add_sep_token=False,
        )
        data = {
            "custom_labels": torch.tensor([0]),
            "custom_bboxes": torch.tensor([[10, 20, 30, 40]]),
        }
        output = serializer.build_output(
            data, label_key="custom_labels", bbox_key="custom_bboxes"
        )
        assert output == "text 10 20 30 40"


@pytest.mark.unit
class TestSerializerHtmlOutput:
    """Test Serializer html output format"""

    def test_html_output_with_index(self):
        serializer = Serializer(
            input_format="seq",
            output_format="html",
            index2label={0: "text", 1: "title"},
            canvas_width=100,
            canvas_height=200,
            add_index_token=True,
            add_sep_token=True,
            add_unk_token=False,
        )
        data = {
            "labels": torch.tensor([0, 1]),
            "discrete_gold_bboxes": torch.tensor([[10, 20, 30, 40], [1, 2, 3, 4]]),
        }
        output = serializer.build_output(data)
        expected = "".join(
            [
                HTML_PREFIX.format(100, 200),
                HTML_TEMPLATE_WITH_INDEX.format("text", 0, 10, 20, 30, 40),
                HTML_TEMPLATE_WITH_INDEX.format("title", 1, 1, 2, 3, 4),
                HTML_SUFFIX,
            ]
        )
        assert output == expected

    def test_html_output_without_index(self):
        serializer = Serializer(
            input_format="seq",
            output_format="html",
            index2label={0: "text"},
            canvas_width=100,
            canvas_height=200,
            add_index_token=False,
            add_sep_token=False,
        )
        data = {
            "labels": torch.tensor([0]),
            "discrete_gold_bboxes": torch.tensor([[10, 20, 30, 40]]),
        }
        output = serializer.build_output(data)
        expected = "".join(
            [
                HTML_PREFIX.format(100, 200),
                HTML_TEMPLATE.format("text", 10, 20, 30, 40),
                HTML_SUFFIX,
            ]
        )
        assert output == expected


@pytest.mark.unit
class TestGenTypeSerializer:
    """Test GenTypeSerializer class"""

    def test_seq_input_with_unk_tokens(self):
        serializer = GenTypeSerializer(
            input_format="seq",
            output_format="seq",
            index2label={0: "text", 1: "title"},
            canvas_width=100,
            canvas_height=200,
            add_index_token=True,
            add_sep_token=True,
            add_unk_token=True,
        )
        data = {"labels": torch.tensor([0, 1])}
        output = serializer.build_input(data)
        assert (
            output
            == "Element Type Constraint: text 0 <unk> <unk> <unk> <unk> | title 1 <unk> <unk> <unk> <unk>"
        )

    def test_seq_input_without_unk(self):
        serializer = GenTypeSerializer(
            input_format="seq",
            output_format="seq",
            index2label={0: "text", 1: "title"},
            canvas_width=100,
            canvas_height=200,
            add_index_token=True,
            add_sep_token=True,
            add_unk_token=False,
        )
        data = {"labels": torch.tensor([0, 1])}
        output = serializer.build_input(data)
        assert output == "Element Type Constraint: text 0 | title 1"

    def test_html_input_with_index_and_unk(self):
        serializer = GenTypeSerializer(
            input_format="html",
            output_format="html",
            index2label={0: "text"},
            canvas_width=100,
            canvas_height=200,
            add_index_token=True,
            add_unk_token=True,
        )
        data = {"labels": torch.tensor([0])}
        output = serializer.build_input(data)
        assert "Element Type Constraint:" in output
        assert 'class="text"' in output
        assert "index: 0" in output
        assert "<unk>" in output

    def test_html_input_without_index_without_unk(self):
        serializer = GenTypeSerializer(
            input_format="html",
            output_format="html",
            index2label={0: "text"},
            canvas_width=100,
            canvas_height=200,
            add_index_token=False,
            add_unk_token=False,
        )
        data = {"labels": torch.tensor([0])}
        output = serializer.build_input(data)
        assert 'class="text"' in output
        assert "index:" not in output
        # Canvas will have left: attribute but not element divs
        assert output.count("left:") == 1  # Only canvas has position


@pytest.mark.unit
class TestGenTypeSizeSerializer:
    """Test GenTypeSizeSerializer class"""

    def test_seq_input_with_size(self):
        serializer = GenTypeSizeSerializer(
            input_format="seq",
            output_format="seq",
            index2label={0: "text", 1: "title"},
            canvas_width=100,
            canvas_height=200,
            add_index_token=True,
            add_sep_token=True,
            add_unk_token=True,
        )
        data = {
            "labels": torch.tensor([0, 1]),
            "discrete_gold_bboxes": torch.tensor([[10, 20, 30, 40], [1, 2, 50, 60]]),
        }
        output = serializer.build_input(data)
        assert (
            output
            == "Element Type and Size Constraint: text 0 <unk> <unk> 30 40 | title 1 <unk> <unk> 50 60"
        )

    def test_seq_input_without_unk(self):
        serializer = GenTypeSizeSerializer(
            input_format="seq",
            output_format="seq",
            index2label={0: "text"},
            canvas_width=100,
            canvas_height=200,
            add_index_token=True,
            add_sep_token=False,
            add_unk_token=False,
        )
        data = {
            "labels": torch.tensor([0]),
            "discrete_gold_bboxes": torch.tensor([[10, 20, 30, 40]]),
        }
        output = serializer.build_input(data)
        assert output == "Element Type and Size Constraint: text 0 30 40"

    def test_html_input_with_size(self):
        serializer = GenTypeSizeSerializer(
            input_format="html",
            output_format="html",
            index2label={0: "text"},
            canvas_width=100,
            canvas_height=200,
            add_index_token=True,
            add_unk_token=False,
        )
        data = {
            "labels": torch.tensor([0]),
            "discrete_gold_bboxes": torch.tensor([[10, 20, 30, 40]]),
        }
        output = serializer.build_input(data)
        assert "width: 30px" in output
        assert "height: 40px" in output


@pytest.mark.unit
class TestGenRelationSerializer:
    """Test GenRelationSerializer class"""

    def test_seq_input_with_relations(self):
        serializer = GenRelationSerializer(
            input_format="seq",
            output_format="seq",
            index2label={1: "text", 2: "image"},
            canvas_width=100,
            canvas_height=200,
            add_index_token=True,
            add_sep_token=True,
            add_unk_token=False,
        )
        data = {
            "labels": torch.tensor([1, 2]),
            "relations": torch.tensor(
                [[0, 0, 1, 0, 3]]
            ),  # canvas, canvas, text, 0, top
        }
        output = serializer.build_input(data)
        assert (
            output
            == "Element Type Constraint: text 0 | image 1\nElement Relationship Constraint: text 0 top canvas"
        )

    def test_seq_input_without_relations(self):
        serializer = GenRelationSerializer(
            input_format="seq",
            output_format="seq",
            index2label={1: "text"},
            canvas_width=100,
            canvas_height=200,
            add_index_token=True,
            add_sep_token=True,
            add_unk_token=False,
        )
        data = {"labels": torch.tensor([1]), "relations": torch.tensor([])}
        output = serializer.build_input(data)
        assert output == "Element Type Constraint: text 0"
        assert "Relationship Constraint" not in output

    def test_seq_input_with_multiple_relations(self):
        serializer = GenRelationSerializer(
            input_format="seq",
            output_format="seq",
            index2label={1: "text", 2: "image"},
            canvas_width=100,
            canvas_height=200,
            add_index_token=True,
            add_sep_token=True,
            add_unk_token=False,
        )
        data = {
            "labels": torch.tensor([1, 2]),
            "relations": torch.tensor(
                [
                    [
                        1,
                        0,
                        2,
                        1,
                        0,
                    ],  # image 1 smaller text 0 (relations are reversed in output)
                    [2, 1, 0, 0, 6],  # canvas left image 1
                ]
            ),
        }
        output = serializer.build_input(data)
        # Relations format: [label_j, idx_j] rel_type [label_i, idx_i]
        assert "image 1 smaller text 0" in output
        assert "canvas left image 1" in output

    def test_html_input_with_relations(self):
        serializer = GenRelationSerializer(
            input_format="html",
            output_format="html",
            index2label={1: "text", 2: "image"},
            canvas_width=100,
            canvas_height=200,
            add_index_token=True,
            add_sep_token=True,
            add_unk_token=False,
        )
        data = {
            "labels": torch.tensor([1, 2]),
            "relations": torch.tensor([[1, 0, 2, 1, 3]]),
        }
        output = serializer.build_input(data)
        assert "Element Type Constraint:" in output
        assert 'class="text"' in output
        assert "Element Relationship Constraint:" in output

    def test_html_input_without_relations(self):
        serializer = GenRelationSerializer(
            input_format="html",
            output_format="html",
            index2label={1: "text"},
            canvas_width=100,
            canvas_height=200,
            add_index_token=True,
            add_unk_token=False,
        )
        data = {"labels": torch.tensor([1]), "relations": torch.tensor([])}
        output = serializer.build_input(data)
        assert "Element Type Constraint:" in output
        assert "Relationship Constraint" not in output


@pytest.mark.unit
class TestCompletionSerializer:
    """Test CompletionSerializer class"""

    def test_seq_input_partial_layout(self):
        serializer = CompletionSerializer(
            input_format="seq",
            output_format="seq",
            index2label={0: "text", 1: "title"},
            canvas_width=100,
            canvas_height=200,
            add_index_token=True,
            add_sep_token=False,
        )
        data = {
            "labels": torch.tensor([0, 1, 1]),
            "discrete_bboxes": torch.tensor(
                [[10, 20, 30, 40], [1, 2, 3, 4], [5, 6, 7, 8]]
            ),
        }
        output = serializer.build_input(data)
        # Should only include first element
        assert output == "Partial Layout: text 0 10 20 30 40"

    def test_html_input_partial_layout(self):
        serializer = CompletionSerializer(
            input_format="html",
            output_format="html",
            index2label={0: "text"},
            canvas_width=100,
            canvas_height=200,
            add_index_token=False,
        )
        data = {
            "labels": torch.tensor([0, 0]),
            "discrete_bboxes": torch.tensor([[10, 20, 30, 40], [50, 60, 70, 80]]),
        }
        output = serializer.build_input(data)
        assert "Partial Layout:" in output
        # Should only contain first element
        assert output.count('class="text"') == 1


@pytest.mark.unit
class TestRefinementSerializer:
    """Test RefinementSerializer class"""

    def test_seq_input_noise_layout(self):
        serializer = RefinementSerializer(
            input_format="seq",
            output_format="seq",
            index2label={0: "text"},
            canvas_width=120,
            canvas_height=160,
            add_index_token=False,
            add_sep_token=True,
            add_unk_token=False,
        )
        data = {
            "labels": torch.tensor([0]),
            "discrete_bboxes": torch.tensor([[10, 20, 30, 40]]),
            "discrete_gold_bboxes": torch.tensor([[11, 21, 31, 41]]),
        }
        output = serializer.build_input(data)
        assert output == "Noise Layout: text 10 20 30 40"

    def test_html_input_noise_layout(self):
        serializer = RefinementSerializer(
            input_format="html",
            output_format="html",
            index2label={0: "text"},
            canvas_width=100,
            canvas_height=200,
            add_index_token=True,
        )
        data = {
            "labels": torch.tensor([0]),
            "discrete_bboxes": torch.tensor([[10, 20, 30, 40]]),
        }
        output = serializer.build_input(data)
        assert "Noise Layout:" in output
        assert 'class="text"' in output


@pytest.mark.unit
class TestContentAwareSerializer:
    """Test ContentAwareSerializer class"""

    def test_seq_input_with_content(self):
        serializer = ContentAwareSerializer(
            input_format="seq",
            output_format="seq",
            index2label={0: "text", 1: "logo"},
            canvas_width=100,
            canvas_height=200,
            add_index_token=True,
            add_sep_token=True,
            add_unk_token=True,
        )
        data = {
            "labels": torch.tensor([0, 1]),
            "discrete_content_bboxes": torch.tensor(
                [[10, 20, 30, 40], [50, 60, 70, 80]]
            ),
        }
        output = serializer.build_input(data)
        assert "Content Constraint:" in output
        assert "left 10px, top 20px, width 30px, height 40px" in output
        assert "Element Type Constraint:" in output
        assert "text 0 <unk> <unk> <unk> <unk>" in output

    def test_seq_input_without_content(self):
        serializer = ContentAwareSerializer(
            input_format="seq",
            output_format="seq",
            index2label={0: "text"},
            canvas_width=100,
            canvas_height=200,
            add_index_token=False,
            add_unk_token=False,
        )
        data = {
            "labels": torch.tensor([0]),
            "discrete_content_bboxes": torch.tensor([]),
        }
        output = serializer.build_input(data)
        assert "Content Constraint:" in output
        assert "Element Type Constraint:" in output


@pytest.mark.unit
class TestTextToLayoutSerializer:
    """Test TextToLayoutSerializer class"""

    def test_seq_input_with_text(self):
        serializer = TextToLayoutSerializer(
            input_format="seq",
            output_format="seq",
            index2label={0: "text"},
            canvas_width=100,
            canvas_height=200,
        )
        data = {"text": "This is a sample text prompt"}
        output = serializer.build_input(data)
        assert output == "Text: This is a sample text prompt"

    def test_seq_input_empty_text(self):
        serializer = TextToLayoutSerializer(
            input_format="seq",
            output_format="seq",
            index2label={0: "text"},
            canvas_width=100,
            canvas_height=200,
        )
        data = {"text": ""}
        output = serializer.build_input(data)
        assert output == "Text: "


@pytest.mark.unit
class TestBuildPrompt:
    """Test build_prompt function"""

    def test_build_prompt_includes_preamble_and_examples(self):
        serializer = RefinementSerializer(
            input_format="seq",
            output_format="seq",
            index2label={0: "text"},
            canvas_width=120,
            canvas_height=160,
            add_index_token=False,
            add_sep_token=True,
            add_unk_token=False,
        )
        exemplar = {
            "labels": torch.tensor([0]),
            "discrete_bboxes": torch.tensor([[10, 20, 30, 40]]),
            "discrete_gold_bboxes": torch.tensor([[10, 20, 30, 40]]),
        }
        test_data = {
            "labels": torch.tensor([0]),
            "discrete_bboxes": torch.tensor([[1, 2, 3, 4]]),
            "discrete_gold_bboxes": torch.tensor([[1, 2, 3, 4]]),
        }
        prompt = build_prompt(
            serializer=serializer,
            exemplars=[exemplar],
            test_data=test_data,
            dataset="publaynet",
        )
        assert prompt.startswith(
            PREAMBLE.format(serializer.task_type, "document", 120, 160)
        )
        assert "Noise Layout:" in prompt
        assert "text 10 20 30 40" in prompt

    def test_build_prompt_with_multiple_exemplars(self):
        serializer = GenTypeSerializer(
            input_format="seq",
            output_format="seq",
            index2label={0: "text", 1: "title"},
            canvas_width=100,
            canvas_height=200,
            add_index_token=False,
            add_sep_token=False,
        )
        exemplars = [
            {
                "labels": torch.tensor([0]),
                "discrete_gold_bboxes": torch.tensor([[10, 20, 30, 40]]),
            },
            {
                "labels": torch.tensor([1]),
                "discrete_gold_bboxes": torch.tensor([[5, 10, 15, 20]]),
            },
        ]
        test_data = {
            "labels": torch.tensor([0, 1]),
            "discrete_gold_bboxes": torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]]),
        }
        prompt = build_prompt(
            serializer=serializer,
            exemplars=exemplars,
            test_data=test_data,
            dataset="rico",
        )
        # Both exemplars should be included
        assert "text 10 20 30 40" in prompt
        assert "title 5 10 15 20" in prompt

    def test_build_prompt_respects_max_length(self):
        serializer = GenTypeSerializer(
            input_format="seq",
            output_format="seq",
            index2label={0: "text"},
            canvas_width=100,
            canvas_height=200,
        )
        # Create many exemplars
        exemplars = [
            {
                "labels": torch.tensor([0] * 10),
                "discrete_gold_bboxes": torch.tensor([[10, 20, 30, 40]] * 10),
            }
            for _ in range(100)
        ]
        test_data = {
            "labels": torch.tensor([0]),
            "discrete_gold_bboxes": torch.tensor([[1, 2, 3, 4]]),
        }
        prompt = build_prompt(
            serializer=serializer,
            exemplars=exemplars,
            test_data=test_data,
            dataset="rico",
            max_length=500,
        )
        # Prompt should be truncated
        assert len(prompt) <= 600  # Some buffer for test data

    def test_build_prompt_custom_separators(self):
        serializer = RefinementSerializer(
            input_format="seq",
            output_format="seq",
            index2label={0: "text"},
            canvas_width=100,
            canvas_height=200,
        )
        exemplar = {
            "labels": torch.tensor([0]),
            "discrete_bboxes": torch.tensor([[10, 20, 30, 40]]),
            "discrete_gold_bboxes": torch.tensor([[10, 20, 30, 40]]),
        }
        test_data = {
            "labels": torch.tensor([0]),
            "discrete_bboxes": torch.tensor([[1, 2, 3, 4]]),
            "discrete_gold_bboxes": torch.tensor([[1, 2, 3, 4]]),
        }
        prompt = build_prompt(
            serializer=serializer,
            exemplars=[exemplar],
            test_data=test_data,
            dataset="publaynet",
            separator_in_samples="###",
            separator_between_samples="---",
        )
        assert "###" in prompt
        assert "---" in prompt


@pytest.mark.unit
class TestCreateSerializer:
    """Test create_serializer factory function"""

    def test_create_gent_serializer(self):
        serializer = create_serializer(
            dataset="rico",
            task="gent",
            input_format="seq",
            output_format="seq",
            add_index_token=True,
            add_sep_token=True,
            add_unk_token=False,
        )
        assert isinstance(serializer, GenTypeSerializer)
        assert serializer.canvas_width == 90
        assert serializer.canvas_height == 160

    def test_create_gents_serializer(self):
        serializer = create_serializer(
            dataset="publaynet",
            task="gents",
            input_format="seq",
            output_format="html",
            add_index_token=True,
            add_sep_token=True,
            add_unk_token=True,
        )
        assert isinstance(serializer, GenTypeSizeSerializer)
        assert serializer.canvas_width == 120
        assert serializer.canvas_height == 160

    def test_create_genr_serializer(self):
        serializer = create_serializer(
            dataset="rico",
            task="genr",
            input_format="html",
            output_format="html",
            add_index_token=True,
            add_sep_token=True,
            add_unk_token=False,
        )
        assert isinstance(serializer, GenRelationSerializer)

    def test_create_completion_serializer(self):
        serializer = create_serializer(
            dataset="publaynet",
            task="completion",
            input_format="seq",
            output_format="seq",
            add_index_token=False,
            add_sep_token=False,
            add_unk_token=False,
        )
        assert isinstance(serializer, CompletionSerializer)

    def test_create_refinement_serializer(self):
        serializer = create_serializer(
            dataset="publaynet",
            task="refinement",
            input_format="seq",
            output_format="html",
            add_index_token=True,
            add_sep_token=True,
            add_unk_token=False,
        )
        assert isinstance(serializer, RefinementSerializer)

    def test_create_content_serializer(self):
        serializer = create_serializer(
            dataset="posterlayout",
            task="content",
            input_format="seq",
            output_format="seq",
            add_index_token=True,
            add_sep_token=True,
            add_unk_token=True,
        )
        assert isinstance(serializer, ContentAwareSerializer)

    def test_create_text_serializer(self):
        serializer = create_serializer(
            dataset="webui",
            task="text",
            input_format="seq",
            output_format="seq",
            add_index_token=False,
            add_sep_token=False,
            add_unk_token=False,
        )
        assert isinstance(serializer, TextToLayoutSerializer)


@pytest.mark.unit
class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_empty_layout(self):
        serializer = Serializer(
            input_format="seq",
            output_format="seq",
            index2label={0: "text"},
            canvas_width=100,
            canvas_height=200,
        )
        data = {"labels": torch.tensor([]), "discrete_gold_bboxes": torch.tensor([])}
        output = serializer.build_output(data)
        assert output == ""

    def test_single_element(self):
        serializer = GenTypeSerializer(
            input_format="seq",
            output_format="seq",
            index2label={0: "text"},
            canvas_width=100,
            canvas_height=200,
            add_index_token=True,
            add_sep_token=True,
        )
        data = {"labels": torch.tensor([0])}
        output = serializer.build_input(data)
        assert "|" not in output  # No separator for single element

    def test_max_elements(self):
        serializer = GenTypeSerializer(
            input_format="seq",
            output_format="seq",
            index2label={i: f"type{i}" for i in range(25)},
            canvas_width=100,
            canvas_height=200,
            add_index_token=True,
            add_sep_token=True,
        )
        # Test with many elements
        data = {"labels": torch.tensor(list(range(25)))}
        output = serializer.build_input(data)
        assert output.count("|") == 24  # 24 separators for 25 elements

    def test_html_special_characters_in_labels(self):
        serializer = Serializer(
            input_format="seq",
            output_format="html",
            index2label={0: "text&special"},
            canvas_width=100,
            canvas_height=200,
        )
        data = {
            "labels": torch.tensor([0]),
            "discrete_gold_bboxes": torch.tensor([[10, 20, 30, 40]]),
        }
        output = serializer.build_output(data)
        assert "text&special" in output

    def test_zero_canvas_dimensions_handled(self):
        # This tests that serializers handle unusual canvas sizes
        serializer = Serializer(
            input_format="seq",
            output_format="seq",
            index2label={0: "text"},
            canvas_width=1,
            canvas_height=1,
        )
        data = {
            "labels": torch.tensor([0]),
            "discrete_gold_bboxes": torch.tensor([[0, 0, 1, 1]]),
        }
        output = serializer.build_output(data)
        assert "text" in output
