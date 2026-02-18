import torch
import pytest

from layoutprompter.parsing import Parser


@pytest.mark.unit
def test_parser_seq_parses_labels_and_bboxes():
    parser = Parser(dataset="publaynet", output_format="seq")
    prediction = "text 12 16 30 40 | title 1 2 3 4"
    labels, bboxes = parser._extract_labels_and_bboxes(prediction)

    assert labels.tolist() == [1, 2]
    expected = torch.tensor(
        [
            [12 / 120, 16 / 160, 30 / 120, 40 / 160],
            [1 / 120, 2 / 160, 3 / 120, 4 / 160],
        ]
    )
    assert torch.allclose(bboxes, expected)


@pytest.mark.unit
def test_parser_html_parses_labels_and_bboxes():
    parser = Parser(dataset="publaynet", output_format="html")
    prediction = (
        "<html>\n<body>\n"
        '<div class="canvas" style="left: 0px; top: 0px; width: 120px; height: 160px"></div>\n'
        '<div class="text" style="left: 12px; top: 16px; width: 30px; height: 40px"></div>\n'
        '<div class="title" style="left: 0px; top: 80px; width: 60px; height: 20px"></div>\n'
        "</body>\n</html>"
    )

    labels, bboxes = parser._extract_labels_and_bboxes(prediction)

    assert labels.tolist() == [1, 2]
    expected = torch.tensor(
        [
            [12 / 120, 16 / 160, 30 / 120, 40 / 160],
            [0 / 120, 80 / 160, 60 / 120, 20 / 160],
        ]
    )
    assert torch.allclose(bboxes, expected)


@pytest.mark.unit
def test_parser_html_raises_on_mismatched_fields():
    parser = Parser(dataset="publaynet", output_format="html")
    invalid = (
        "<html>\n<body>\n"
        '<div class="canvas" style="left: 0px; top: 0px; width: 120px; height: 160px"></div>\n'
        '<div class="text" style="left: 12px; top: 16px"></div>\n'
        "</body>\n</html>"
    )

    with pytest.raises(RuntimeError):
        parser._extract_labels_and_bboxes(invalid)
