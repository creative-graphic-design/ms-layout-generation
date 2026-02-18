import argparse

from layoutformer_pp.utils import config


def test_add_arguments_defaults_and_flags():
    parser = argparse.ArgumentParser()
    config.add_arguments(parser)

    args = parser.parse_args([])
    assert args.data_dir == "../datasets/"
    assert args.out_dir == "../output/"
    assert args.batch_size == 8
    assert args.eval_batch_size == 16

    args = parser.parse_args(["--test"])
    assert args.test is True

    args = parser.parse_args(["--train"])
    assert args.test is False


def test_add_task_arguments_includes_eval_seed():
    parser = argparse.ArgumentParser()
    config.add_task_arguments(parser)

    args = parser.parse_args(["--eval_seed", "123"])
    assert args.eval_seed == 123
    assert args.tasks == "refinement"
    assert args.add_sep_token is False
    assert args.add_task_prompt is False


def test_add_trainer_arguments_parses_defaults():
    parser = argparse.ArgumentParser()
    config.add_trainer_arguments(parser)

    args = parser.parse_args([])
    assert args.trainer == "basic"
    assert args.backend == "nccl"
    assert args.local_rank == 0
