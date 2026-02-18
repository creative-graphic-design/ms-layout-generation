from layoutformer_pp.utils import os_utils


def test_makedirs_idempotent(tmp_path):
    target = tmp_path / "a" / "b"

    os_utils.makedirs(target)
    assert target.is_dir()

    os_utils.makedirs(target)
    assert target.is_dir()


def test_files_exist_true_and_false(tmp_path):
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"

    first.write_text("first")
    second.write_text("second")

    assert os_utils.files_exist([first, second])

    second.unlink()
    assert not os_utils.files_exist([first, second])
