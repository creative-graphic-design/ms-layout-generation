from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch

from improved_diffusion.text_datasets import (
    TextDataset,
    _collate_batch_helper,
    get_corpus_rocstory,
    helper_tokenize_encode,
    load_data_text,
)


class MockDataArgs:
    """Mock data_args object for testing"""

    def __init__(
        self,
        checkpoint_path: str,
        experiment: str = "random",
        experiment_mode: str = "lm",
        model_arch: str = "transformer",
        modality: str = "e2e-tgt",
        training_mode: str = "discrete1",
        in_channel: int = 8,
        e2e_train: str = "",
        noise_level: float = 0.0,
    ):
        self.checkpoint_path = checkpoint_path
        self.experiment = experiment
        self.experiment_mode = experiment_mode
        self.model_arch = model_arch
        self.modality = modality
        self.training_mode = training_mode
        self.in_channel = in_channel
        self.e2e_train = e2e_train
        self.noise_level = noise_level


class TestCollateBatchHelper:
    """Test _collate_batch_helper function"""

    def test_collate_basic(self) -> None:
        """Test basic padding functionality"""
        examples = [[1, 2, 3], [4, 5], [6]]
        pad_token_id = 0
        max_length = 5

        result = _collate_batch_helper(examples, pad_token_id, max_length)

        assert len(result) == 3
        assert len(result[0]) == 5
        assert result[0] == [1, 2, 3, 0, 0]
        assert result[1] == [4, 5, 0, 0, 0]
        assert result[2] == [6, 0, 0, 0, 0]

    def test_collate_with_truncation(self) -> None:
        """Test that sequences longer than max_length are truncated"""
        examples = [[1, 2, 3, 4, 5, 6, 7], [8, 9]]
        pad_token_id = 0
        max_length = 5

        result = _collate_batch_helper(examples, pad_token_id, max_length)

        assert len(result[0]) == 5
        assert result[0] == [1, 2, 3, 4, 5]
        assert result[1] == [8, 9, 0, 0, 0]

    def test_collate_with_mask(self) -> None:
        """Test collate helper with return_mask=True"""
        examples = [[1, 2, 3], [4, 5], [6]]
        pad_token_id = 0
        max_length = 5

        result, mask = _collate_batch_helper(
            examples, pad_token_id, max_length, return_mask=True
        )

        assert len(result) == 3
        assert len(mask) == 3
        assert mask[0] == [1, 1, 1, 0, 0]
        assert mask[1] == [1, 1, 0, 0, 0]
        assert mask[2] == [1, 0, 0, 0, 0]

    def test_collate_empty_sequences(self) -> None:
        """Test collate with empty sequences"""
        examples = [[], [1], []]
        pad_token_id = 999
        max_length = 3

        result = _collate_batch_helper(examples, pad_token_id, max_length)

        assert result[0] == [999, 999, 999]
        assert result[1] == [1, 999, 999]
        assert result[2] == [999, 999, 999]


class TestHelperTokenizeEncode:
    """Test helper_tokenize_encode function"""

    def test_helper_tokenize_encode_basic(self, tmp_path: Path) -> None:
        """Test basic tokenization and encoding"""
        sentence_lst = [
            ["hello", "world"],
            ["test", "sequence"],
        ]

        vocab_dict = {
            "START": 0,
            "END": 1,
            "UNK": 2,
            "PAD": 3,
            "|": 4,
            "hello": 5,
            "world": 6,
            "test": 7,
            "sequence": 8,
        }

        model = torch.nn.Embedding(len(vocab_dict), 8)
        torch.nn.init.normal_(model.weight)

        data_args = MockDataArgs(
            checkpoint_path=str(tmp_path),
            experiment="random",
            e2e_train="test_data",
        )

        result = helper_tokenize_encode(
            sentence_lst,
            vocab_dict,
            model,
            seqlen=10,
            data_args=data_args,
            padding_mode="pad",
        )

        assert len(result) == 2
        assert "input_ids" in result[0]
        assert "hidden_states" in result[0]
        assert len(result[0]["input_ids"]) == 10

    def test_helper_tokenize_encode_with_unk(self, tmp_path: Path) -> None:
        """Test tokenization with unknown tokens"""
        sentence_lst = [["known", "unknown"]]

        vocab_dict = {"START": 0, "END": 1, "UNK": 2, "PAD": 3, "|": 4, "known": 5}

        model = torch.nn.Embedding(len(vocab_dict), 8)

        data_args = MockDataArgs(
            checkpoint_path=str(tmp_path),
            experiment="random",
            e2e_train="test_data",
        )

        result = helper_tokenize_encode(
            sentence_lst,
            vocab_dict,
            model,
            seqlen=10,
            data_args=data_args,
            padding_mode="pad",
        )

        # Check that unknown token was replaced with UNK
        assert result[0]["input_ids"][0] == 0  # START
        assert result[0]["input_ids"][1] == 5  # known
        assert result[0]["input_ids"][2] == 2  # UNK (for unknown)
        assert result[0]["input_ids"][3] == 1  # END

    def test_helper_tokenize_encode_block_mode(self, tmp_path: Path) -> None:
        """Test tokenization with block padding mode"""
        sentence_lst = [
            ["a", "b", "c"],
            ["d", "e"],
        ]

        vocab_dict = {
            "START": 0,
            "END": 1,
            "UNK": 2,
            "PAD": 3,
            "|": 4,
            "a": 5,
            "b": 6,
            "c": 7,
            "d": 8,
            "e": 9,
        }

        model = torch.nn.Embedding(len(vocab_dict), 8)

        data_args = MockDataArgs(
            checkpoint_path=str(tmp_path),
            experiment="random",
            e2e_train="test_data",
        )

        result = helper_tokenize_encode(
            sentence_lst,
            vocab_dict,
            model,
            seqlen=5,
            data_args=data_args,
            padding_mode="block",
        )

        # Block mode concatenates and splits into fixed-size blocks
        assert all("input_ids" in item for item in result)


class TestGetCorpusRocstory:
    """Test get_corpus_rocstory function with real data"""

    def test_get_corpus_train_split(
        self, small_rico_dataset: Path, tmp_path: Path
    ) -> None:
        """Test loading train split of RICO dataset"""
        data_args = MockDataArgs(
            checkpoint_path=str(tmp_path),
            experiment="random",
            experiment_mode="lm",
            modality="e2e-tgt",
            e2e_train=str(small_rico_dataset),
        )

        result, model = get_corpus_rocstory(
            data_args, model=None, seq_length=121, padding_mode="pad", split="train"
        )

        assert "train" in result
        assert len(result["train"]) > 0
        assert model is not None

        # Check vocab was created
        vocab_path = tmp_path / "vocab.json"
        assert vocab_path.exists()

        with open(vocab_path) as f:
            vocab = json.load(f)
            assert "PAD" in vocab
            assert "START" in vocab or len(vocab) > 0

    def test_get_corpus_valid_split(
        self, small_rico_dataset: Path, tmp_path: Path
    ) -> None:
        """Test loading valid split"""
        data_args = MockDataArgs(
            checkpoint_path=str(tmp_path),
            experiment="random",
            experiment_mode="lm",
            modality="e2e-tgt",
            e2e_train=str(small_rico_dataset),
        )

        result, model = get_corpus_rocstory(
            data_args, model=None, seq_length=121, padding_mode="pad", split="valid"
        )

        assert "train" in result
        assert len(result["train"]) > 0

    def test_get_corpus_test_split(
        self, small_rico_dataset: Path, tmp_path: Path
    ) -> None:
        """Test loading test split"""
        processed_root = (
            tmp_path / "data" / "processed_datasets" / Path(small_rico_dataset).name
        )
        processed_root.mkdir(parents=True, exist_ok=True)
        src_test = Path(small_rico_dataset) / "src1_test.txt"
        if src_test.exists():
            (processed_root / "src1_test.txt").write_text(
                src_test.read_text(encoding="utf-8"), encoding="utf-8"
            )

        data_args = MockDataArgs(
            checkpoint_path=str(tmp_path),
            experiment="random",
            experiment_mode="lm",
            modality="e2e-tgt",
            e2e_train=str(small_rico_dataset),
        )

        work_dir = tmp_path / "work"
        work_dir.mkdir(parents=True, exist_ok=True)
        old_cwd = os.getcwd()
        try:
            os.chdir(work_dir)
            result, model = get_corpus_rocstory(
                data_args, model=None, seq_length=121, padding_mode="pad", split="test"
            )
        finally:
            os.chdir(old_cwd)

        assert "train" in result
        assert len(result["train"]) > 0

    def test_get_corpus_with_existing_vocab(
        self, small_rico_dataset: Path, tmp_path: Path
    ) -> None:
        """Test loading corpus with pre-existing vocab"""
        # First create vocab
        data_args = MockDataArgs(
            checkpoint_path=str(tmp_path),
            experiment="random",
            experiment_mode="lm",
            modality="e2e-tgt",
            e2e_train=str(small_rico_dataset),
        )

        result1, model1 = get_corpus_rocstory(
            data_args, model=None, seq_length=121, padding_mode="pad", split="train"
        )

        vocab_path = tmp_path / "vocab.json"
        with open(vocab_path) as f:
            vocab = json.load(f)

        # Now load with existing vocab
        result2, model2 = get_corpus_rocstory(
            data_args,
            model=model1,
            seq_length=121,
            padding_mode="pad",
            split="train",
            load_vocab=vocab,
        )

        assert len(result1["train"]) == len(result2["train"])


class TestTextDataset:
    """Test TextDataset class"""

    def test_text_dataset_initialization(self, tmp_path: Path) -> None:
        """Test TextDataset initialization"""
        text_datasets = {
            "train": [
                {
                    "input_ids": [0, 1, 2, 3],
                    "hidden_states": [[0.1] * 8] * 4,
                },
                {
                    "input_ids": [4, 5, 6, 7],
                    "hidden_states": [[0.2] * 8] * 4,
                },
            ]
        }

        data_args = MockDataArgs(checkpoint_path=str(tmp_path))

        dataset = TextDataset(
            text_datasets=text_datasets,
            resolution=121,
            data_args=data_args,
            model_arch="transformer",
        )

        assert len(dataset) == 2
        assert dataset.resolution == 121

    def test_text_dataset_getitem(self, tmp_path: Path) -> None:
        """Test TextDataset __getitem__ method"""
        text_datasets = {
            "train": [
                {
                    "input_ids": [0, 1, 2, 3],
                    "hidden_states": [[0.1, 0.2, 0.3, 0.4] * 2] * 4,
                }
            ]
        }

        data_args = MockDataArgs(checkpoint_path=str(tmp_path))

        dataset = TextDataset(
            text_datasets=text_datasets,
            resolution=121,
            data_args=data_args,
            model_arch="transformer",
        )

        arr, out_dict = dataset[0]

        assert isinstance(arr, np.ndarray)
        assert "input_ids" in out_dict
        assert isinstance(out_dict["input_ids"], np.ndarray)

    def test_text_dataset_with_noise(self, tmp_path: Path) -> None:
        """Test TextDataset with noise_level > 0"""
        text_datasets = {
            "train": [
                {
                    "input_ids": [0, 1, 2, 3],
                    "hidden_states": [[0.1] * 8] * 4,
                }
            ]
        }

        data_args = MockDataArgs(checkpoint_path=str(tmp_path), noise_level=0.1)

        dataset = TextDataset(
            text_datasets=text_datasets,
            resolution=121,
            data_args=data_args,
            model_arch="transformer",
        )

        arr1, _ = dataset[0]
        arr2, _ = dataset[0]

        # With noise, same sample should give different results
        assert not np.allclose(arr1, arr2)

    def test_text_dataset_sharding(self, tmp_path: Path) -> None:
        """Test TextDataset with multi-process sharding"""
        train_samples = [
            {"input_ids": [i], "hidden_states": [[float(i)] * 8] * 1} for i in range(10)
        ]

        data_args = MockDataArgs(checkpoint_path=str(tmp_path))

        # Create dataset with shard 0 of 2
        dataset_shard0 = TextDataset(
            text_datasets={"train": list(train_samples)},
            resolution=121,
            data_args=data_args,
            model_arch="transformer",
            shard=0,
            num_shards=2,
        )

        # Create dataset with shard 1 of 2
        dataset_shard1 = TextDataset(
            text_datasets={"train": list(train_samples)},
            resolution=121,
            data_args=data_args,
            model_arch="transformer",
            shard=1,
            num_shards=2,
        )

        # Each shard should have half the data
        assert len(dataset_shard0) == 5
        assert len(dataset_shard1) == 5

    def test_text_dataset_conditional_gen(self, tmp_path: Path) -> None:
        """Test TextDataset with conditional generation mode"""
        text_datasets = {
            "train": [
                {
                    "input_ids": [0, 1, 2, 3],
                    "hidden_states": [[0.1] * 8] * 4,
                    "src_ids": [5, 6, 7],
                    "src_mask": [1, 1, 1],
                }
            ]
        }

        data_args = MockDataArgs(
            checkpoint_path=str(tmp_path), experiment_mode="conditional_gen"
        )

        dataset = TextDataset(
            text_datasets=text_datasets,
            resolution=121,
            data_args=data_args,
            model_arch="transformer",
        )

        arr, out_dict = dataset[0]

        assert "input_ids" in out_dict
        assert "src_ids" in out_dict
        assert "src_mask" in out_dict


class TestLoadDataText:
    """Test load_data_text function (integration)"""

    def test_load_data_text_basic(
        self, small_rico_dataset: Path, tmp_path: Path
    ) -> None:
        """Test load_data_text with real RICO dataset"""
        data_args = MockDataArgs(
            checkpoint_path=str(tmp_path),
            experiment="random",
            experiment_mode="lm",
            modality="e2e-tgt",
            e2e_train=str(small_rico_dataset),
        )

        data_loader = load_data_text(
            data_dir=str(small_rico_dataset),
            batch_size=2,
            seq_length=121,
            class_cond=False,
            deterministic=True,
            data_args=data_args,
            task_mode="e2e-tgt",
            model=None,
            padding_mode="pad",
            split="train",
        )

        # Get first batch
        batch, cond = next(data_loader)

        assert batch.shape[0] == 2  # batch_size
        assert "input_ids" in cond
        assert cond["input_ids"].shape[0] == 2

    def test_load_data_text_deterministic(
        self, small_rico_dataset: Path, tmp_path: Path
    ) -> None:
        """Test deterministic data loading"""
        data_args = MockDataArgs(
            checkpoint_path=str(tmp_path),
            experiment="random",
            experiment_mode="lm",
            modality="e2e-tgt",
            e2e_train=str(small_rico_dataset),
        )

        data_loader1 = load_data_text(
            data_dir=str(small_rico_dataset),
            batch_size=2,
            seq_length=121,
            class_cond=False,
            deterministic=True,
            data_args=data_args,
            task_mode="e2e-tgt",
            model=None,
            padding_mode="pad",
            split="train",
        )

        data_loader2 = load_data_text(
            data_dir=str(small_rico_dataset),
            batch_size=2,
            seq_length=121,
            class_cond=False,
            deterministic=True,
            data_args=data_args,
            task_mode="e2e-tgt",
            model=None,
            padding_mode="pad",
            split="train",
        )

        batch1, cond1 = next(data_loader1)
        batch2, cond2 = next(data_loader2)

        # With deterministic=True and same seed, should get same data
        assert torch.allclose(batch1, batch2)
        assert torch.equal(cond1["input_ids"], cond2["input_ids"])

    def test_load_data_text_non_deterministic(
        self, small_rico_dataset: Path, tmp_path: Path
    ) -> None:
        """Test non-deterministic (shuffled) data loading"""
        data_args = MockDataArgs(
            checkpoint_path=str(tmp_path),
            experiment="random",
            experiment_mode="lm",
            modality="e2e-tgt",
            e2e_train=str(small_rico_dataset),
        )

        data_loader = load_data_text(
            data_dir=str(small_rico_dataset),
            batch_size=2,
            seq_length=121,
            class_cond=False,
            deterministic=False,  # Shuffling enabled
            data_args=data_args,
            task_mode="e2e-tgt",
            model=None,
            padding_mode="pad",
            split="train",
        )

        batch, cond = next(data_loader)

        assert batch.shape[0] == 2
        assert "input_ids" in cond

    def test_load_data_text_infinite_generator(
        self, small_rico_dataset: Path, tmp_path: Path
    ) -> None:
        """Test that data loader works as infinite generator"""
        data_args = MockDataArgs(
            checkpoint_path=str(tmp_path),
            experiment="random",
            experiment_mode="lm",
            modality="e2e-tgt",
            e2e_train=str(small_rico_dataset),
        )

        data_loader = load_data_text(
            data_dir=str(small_rico_dataset),
            batch_size=2,
            seq_length=121,
            class_cond=False,
            deterministic=True,
            data_args=data_args,
            task_mode="e2e-tgt",
            model=None,
            padding_mode="pad",
            split="train",
        )

        # Should be able to get multiple batches
        batches = []
        for _ in range(3):
            batch, cond = next(data_loader)
            batches.append(batch)

        assert len(batches) == 3

    def test_load_data_text_different_batch_sizes(
        self, small_rico_dataset: Path, tmp_path: Path
    ) -> None:
        """Test load_data_text with different batch sizes"""
        data_args = MockDataArgs(
            checkpoint_path=str(tmp_path),
            experiment="random",
            experiment_mode="lm",
            modality="e2e-tgt",
            e2e_train=str(small_rico_dataset),
        )

        for batch_size in [1, 2, 4]:
            data_loader = load_data_text(
                data_dir=str(small_rico_dataset),
                batch_size=batch_size,
                seq_length=121,
                class_cond=False,
                deterministic=True,
                data_args=data_args,
                task_mode="e2e-tgt",
                model=None,
                padding_mode="pad",
                split="train",
            )

            batch, cond = next(data_loader)
            assert batch.shape[0] == batch_size


class TestTextDatasetEdgeCases:
    """Test edge cases and error conditions"""

    def test_empty_sequence_handling(self, tmp_path: Path) -> None:
        """Test handling of empty sequences"""
        text_datasets = {
            "train": [
                {
                    "input_ids": [],
                    "hidden_states": [],
                }
            ]
        }

        data_args = MockDataArgs(checkpoint_path=str(tmp_path))

        dataset = TextDataset(
            text_datasets=text_datasets,
            resolution=121,
            data_args=data_args,
            model_arch="transformer",
        )

        # Should not crash on empty sequence
        arr, out_dict = dataset[0]
        assert arr.shape[0] == 0

    def test_max_length_handling(self, tmp_path: Path) -> None:
        """Test that sequences are properly truncated to max_length"""
        long_sequence = list(range(200))
        text_datasets = {
            "train": [
                {
                    "input_ids": long_sequence,
                    "hidden_states": [[0.1] * 8] * 200,
                }
            ]
        }

        data_args = MockDataArgs(checkpoint_path=str(tmp_path))

        dataset = TextDataset(
            text_datasets=text_datasets,
            resolution=121,  # Max length
            data_args=data_args,
            model_arch="transformer",
        )

        arr, out_dict = dataset[0]

        # Should be truncated to resolution
        assert arr.shape[0] == 200  # Hidden states not truncated in __getitem__
        # But input_ids should respect max_length in tokenization
