import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))


@dataclass
class DummyBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor


class DummyTokenizer:
    pad_token_id = 0

    def __call__(self, text: str, return_tensors: str = None) -> DummyBatch:
        tokens = text.split()
        if tokens:
            ids = torch.arange(1, len(tokens) + 1, dtype=torch.long)
            mask = torch.ones(len(tokens), dtype=torch.long)
        else:
            ids = torch.tensor([self.pad_token_id], dtype=torch.long)
            mask = torch.tensor([0], dtype=torch.long)
        if return_tensors == "pt":
            ids = ids.unsqueeze(0)
            mask = mask.unsqueeze(0)
        return DummyBatch(input_ids=ids, attention_mask=mask)


@pytest.fixture(scope="session")
def parse_then_place_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def rico_stage1_val_path(parse_then_place_root: Path) -> Path:
    return (
        parse_then_place_root
        / "Parse-Then-Place-data"
        / "data"
        / "rico"
        / "stage1"
        / "val.jsonl"
    )


@pytest.fixture(scope="session")
def sample_rico_stage1_example(rico_stage1_val_path: Path) -> dict:
    with rico_stage1_val_path.open(encoding="utf8") as handle:
        line = handle.readline().strip()
    return json.loads(line)


@pytest.fixture()
def dummy_tokenizer() -> DummyTokenizer:
    return DummyTokenizer()
