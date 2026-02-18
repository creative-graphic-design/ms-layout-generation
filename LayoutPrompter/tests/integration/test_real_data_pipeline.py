import os
from pathlib import Path

import pandas as pd
import pytest

from layoutprompter.parsing import Parser
from layoutprompter.preprocess import create_processor
from layoutprompter.ranker import Ranker
from layoutprompter.selection import create_selector
from layoutprompter.serialization import build_prompt, create_serializer
from layoutprompter.utils import RAW_DATA_PATH, read_pt
from layoutprompter.visualization import ContentAwareVisualizer, Visualizer


def _load_raw_pt(dataset_root: Path, dataset: str, split: str):
    path = dataset_root / dataset / "raw" / f"{split}.pt"
    assert path.exists(), f"Missing raw dataset file: {path}"
    return read_pt(str(path))


@pytest.mark.integration
@pytest.mark.real_data
def test_pipeline_gent_rico(dataset_root: Path):
    dataset = "rico"
    train_raw = _load_raw_pt(dataset_root, dataset, "train")
    test_raw = _load_raw_pt(dataset_root, dataset, "test")

    processor = create_processor(dataset=dataset, task="gent")
    processed_train = [processor(item) for item in train_raw[:3]]
    test_data = processor(test_raw[0])

    selector = create_selector("gent", processed_train, candidate_size=-1, num_prompt=2)
    exemplars = selector(test_data)

    serializer = create_serializer(
        dataset=dataset,
        task="gent",
        input_format="seq",
        output_format="html",
        add_index_token=True,
        add_sep_token=True,
        add_unk_token=False,
    )

    prompt = build_prompt(serializer, exemplars, test_data, dataset)
    assert "Task Description" in prompt

    prediction = serializer.build_output(test_data)
    parser = Parser(dataset=dataset, output_format="html")
    parsed = [parser._extract_labels_and_bboxes(prediction)]
    assert parsed

    val_path = dataset_root / dataset / "raw" / "val.pt"
    ranker = Ranker(val_path=str(val_path))
    ranked = ranker(parsed)
    assert ranked

    visualizer = Visualizer(dataset=dataset, times=1)
    images = visualizer(ranked)
    assert images


@pytest.mark.integration
@pytest.mark.real_data
def test_pipeline_genr_publaynet(dataset_root: Path):
    dataset = "publaynet"
    train_raw = _load_raw_pt(dataset_root, dataset, "train")
    test_raw = _load_raw_pt(dataset_root, dataset, "test")

    processor = create_processor(dataset=dataset, task="genr")
    processed_train = [processor(item) for item in train_raw[:3]]
    test_data = processor(test_raw[0])

    selector = create_selector("genr", processed_train, candidate_size=-1, num_prompt=2)
    exemplars = selector(test_data)

    serializer = create_serializer(
        dataset=dataset,
        task="genr",
        input_format="seq",
        output_format="html",
        add_index_token=True,
        add_sep_token=True,
        add_unk_token=False,
    )

    prompt = build_prompt(serializer, exemplars, test_data, dataset)
    assert "Task Description" in prompt

    prediction = serializer.build_output(test_data)
    parser = Parser(dataset=dataset, output_format="html")
    parsed = [parser._extract_labels_and_bboxes(prediction)]
    assert parsed

    val_path = dataset_root / dataset / "raw" / "val.pt"
    ranker = Ranker(val_path=str(val_path))
    ranked = ranker(parsed)
    assert ranked

    visualizer = Visualizer(dataset=dataset, times=1)
    images = visualizer(ranked)
    assert images


@pytest.mark.integration
@pytest.mark.real_data
def test_pipeline_content_posterlayout(dataset_root: Path):
    dataset = "posterlayout"
    raw_root = Path(RAW_DATA_PATH(dataset))
    metadata_path = raw_root / "train_csv_9973.csv"
    assert metadata_path.exists()

    metadata = pd.read_csv(metadata_path)
    processor = create_processor(dataset=dataset, task="content", metadata=metadata)

    train_dir = raw_root / "train" / "saliencymaps_pfpn"
    test_dir = raw_root / "test" / "saliencymaps_pfpn"
    assert train_dir.exists()
    assert test_dir.exists()

    def _collect_samples(directory: Path, split: str, limit: int = 3):
        samples = []
        files = sorted(directory.iterdir(), key=lambda x: int(x.name.split("_")[0]))
        for file in files:
            idx = int(file.name.split("_")[0])
            data = processor(str(file), idx, split)
            if data is not None:
                samples.append(data)
            if len(samples) >= limit:
                break
        return samples

    processed_train = _collect_samples(train_dir, "train", limit=3)
    assert processed_train

    processed_test_candidates = _collect_samples(test_dir, "test", limit=1)
    assert processed_test_candidates
    test_data = processed_test_candidates[0]

    selector = create_selector(
        "content", processed_train, candidate_size=-1, num_prompt=2
    )
    exemplars = selector(test_data)

    serializer = create_serializer(
        dataset=dataset,
        task="content",
        input_format="seq",
        output_format="html",
        add_index_token=True,
        add_sep_token=True,
        add_unk_token=False,
    )

    prompt = build_prompt(serializer, exemplars, test_data, dataset)
    assert "Content Constraint" in prompt

    prediction = serializer.build_output(test_data)
    parser = Parser(dataset=dataset, output_format="html")
    parsed = [parser._extract_labels_and_bboxes(prediction)]
    assert parsed

    ranker = Ranker()
    ranked = ranker(parsed)
    assert ranked

    visualizer = ContentAwareVisualizer(times=1)
    images = visualizer(ranked, test_idx=test_data["idx"])
    assert images
