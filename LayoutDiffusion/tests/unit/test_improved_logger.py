from __future__ import annotations

from pathlib import Path

from improved_diffusion import logger


def test_logger_outputs(tmp_path: Path) -> None:
    logger.configure(dir=str(tmp_path), format_strs=["log", "json", "csv"])
    logger.logkv("loss", 1.0)
    logger.dumpkvs()

    assert (tmp_path / "log.txt").exists()
    assert (tmp_path / "progress.json").exists()
    assert (tmp_path / "progress.csv").exists()

    logger.reset()
