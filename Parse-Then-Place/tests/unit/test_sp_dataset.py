import json

from parse_then_place.semantic_parser.dataset.dataset import SPDataset
from parse_then_place.semantic_parser.dataset.ir_processor import IRProcessor
from parse_then_place.semantic_parser.dataset.text_processor import TextProcessor


def test_sp_dataset_loads_real_example(
    dummy_tokenizer, parse_then_place_root, sample_rico_stage1_example
):
    dataset_root = (
        parse_then_place_root
        / "Parse-Then-Place-data"
        / "data"
        / "rico"
        / "stage1"
    )
    dataset = SPDataset(
        root=str(dataset_root),
        split="val",
        tokenizer=dummy_tokenizer,
        ir_processor=IRProcessor(),
        text_processor=TextProcessor(),
    )
    assert len(dataset) > 0
    sample = dataset[0]
    assert sample["ex_id"] == sample_rico_stage1_example["region_id"]
    assert sample["type"] == sample_rico_stage1_example["region_type"]
    assert sample["value_map"] is None
    assert "el:" not in sample["logical_form"]
    assert "attr:" not in sample["logical_form"]
    assert "element :" in sample["logical_form"]
    assert sample["text_ids"].shape[0] > 0
    assert sample["lf_ids"].shape[0] > 0


def test_sp_dataset_creates_splits_and_handles_ir_variants(tmp_path, dummy_tokenizer):
    dataset_root = tmp_path / "stage1"
    dataset_root.mkdir()
    examples = [
        {
            "region_id": "r0.0",
            "region_type": "header",
            "text": "Example one",
            "ir": ["[el:text button]", "[el:text]"],
        },
        {
            "region_id": "r0.1",
            "region_type": "header",
            "text": "Example two",
            "ir": "[el:image]",
        },
        {
            "region_id": "r0.2",
            "region_type": "header",
            "text": "Missing ir",
        },
    ]
    all_path = dataset_root / "all.jsonl"
    with all_path.open("w", encoding="utf8") as handle:
        for item in examples:
            handle.write(f"{json.dumps(item)}\n")

    dataset = SPDataset(
        root=str(dataset_root),
        split="test",
        tokenizer=dummy_tokenizer,
        ir_processor=IRProcessor(),
        text_processor=TextProcessor(),
    )

    assert (dataset_root / "train.jsonl").exists()
    assert (dataset_root / "val.jsonl").exists()
    assert (dataset_root / "test.jsonl").exists()
    assert len(dataset) == 2
    sample_by_id = {sample["ex_id"]: sample for sample in dataset}
    assert "r0.0" in sample_by_id
    assert "text button" not in sample_by_id["r0.0"]["logical_form"]


def test_sp_dataset_reads_json_split(tmp_path, dummy_tokenizer):
    dataset_root = tmp_path / "stage1"
    dataset_root.mkdir()
    examples = [
        {
            "region_id": "r1.0",
            "region_type": "header",
            "text": "Simple text",
            "ir": "[el:text]",
        }
    ]
    json_path = dataset_root / "val.json"
    json_path.write_text(json.dumps(examples), encoding="utf8")

    dataset = SPDataset(
        root=str(dataset_root),
        split="val",
        tokenizer=dummy_tokenizer,
        ir_processor=IRProcessor(),
        text_processor=TextProcessor(),
    )
    assert len(dataset) == 1
