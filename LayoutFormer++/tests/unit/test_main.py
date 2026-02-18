from types import SimpleNamespace
import importlib
import sys
import types

import torch


def _install_deepspeed_stub():
    if "deepspeed.runtime.lr_schedules" in sys.modules:
        return
    deepspeed_mod = types.ModuleType("deepspeed")
    runtime_mod = types.ModuleType("deepspeed.runtime")
    lr_mod = types.ModuleType("deepspeed.runtime.lr_schedules")

    class DummyWarmupLR:
        def __init__(self, optimizer, warmup_max_lr, warmup_num_steps):
            self.optimizer = optimizer

        def step(self):
            return None

    lr_mod.WarmupLR = DummyWarmupLR
    deepspeed_mod.runtime = runtime_mod
    runtime_mod.lr_schedules = lr_mod

    sys.modules["deepspeed"] = deepspeed_mod
    sys.modules["deepspeed.runtime"] = runtime_mod
    sys.modules["deepspeed.runtime.lr_schedules"] = lr_mod


def test_main_train_and_inference_flow(tmp_path, monkeypatch):
    _install_deepspeed_stub()

    main = importlib.import_module("layoutformer_pp.main")
    main = importlib.reload(main)

    class DummyTokenizer:
        bos_token_id = 0
        pad_token_id = 0
        eos_token_id = 1

        def __len__(self):
            return 10

        def from_vocab(self, path):
            return None

    class DummyDataset:
        def __init__(self, tasks):
            self.tasks = tasks
            self.seq_processor = SimpleNamespace(index2label={1: "label_1"})
            self.colors = [(0, 0, 0)]

        def __len__(self):
            return 1

        def switch_task(self, task):
            self._task = task

    class DummyTrainer:
        last_instance = None

        def __init__(self, *args, **kwargs):
            DummyTrainer.last_instance = self
            self.called = False
            self.cleaned = False

        def __call__(self, train_fn, eval_fn, tasks, eval_interval=1):
            self.called = True
            self.tasks = tasks

        def clean_up(self):
            self.cleaned = True

    class DummyGenerator:
        last_instance = None

        def __init__(self, *args, **kwargs):
            DummyGenerator.last_instance = self
            self.calls = []

        def switch_task(self, task, saved_layouts=None):
            self.calls.append(("switch", task))

        def __call__(self, inference_fn, draw_colors=None, constraint_fn=None):
            self.calls.append(("run", inference_fn))

        def clean_up(self):
            self.calls.append(("cleanup", None))

    class DummyConstraint:
        def __init__(self, *args, **kwargs):
            return None

    dummy_tokenizer = DummyTokenizer()

    monkeypatch.setattr(
        main, "create_tokenizer", lambda *args, **kwargs: dummy_tokenizer
    )
    monkeypatch.setattr(
        main, "build_model", lambda *args, **kwargs: torch.nn.Linear(1, 1)
    )
    monkeypatch.setattr(
        main,
        "create_dataset",
        lambda args, **kwargs: DummyDataset(args.tasks.split(",")),
    )
    monkeypatch.setattr(main, "get_trainer", lambda args: DummyTrainer)
    monkeypatch.setattr(main, "create_fid_model", lambda *args, **kwargs: object())
    monkeypatch.setattr(main, "Generator", DummyGenerator)
    monkeypatch.setattr(
        main.constrained_decoding,
        "TransformerSortByDictLabelConstraint",
        DummyConstraint,
    )
    monkeypatch.setattr(
        main.constrained_decoding,
        "TransformerSortByDictLabelSizeConstraint",
        DummyConstraint,
    )
    monkeypatch.setattr(
        main.constrained_decoding,
        "TransformerSortByDictRelationConstraint",
        DummyConstraint,
    )

    train_args = SimpleNamespace(
        tasks="refinement,completion",
        task_weights=None,
        task_loss_weights="0.5,0.5",
        gaussian_noise_mean=0.0,
        gaussian_noise_std=0.0,
        train_bernoulli_beta=0.0,
        add_sep_token=False,
        add_task_prompt=False,
        num_layers=1,
        nhead=1,
        d_model=4,
        dropout=0.0,
        share_embedding=False,
        sort_by_dict=False,
        partition_training_data=False,
        partition_training_data_task_buckets=None,
        single_task_per_batch=False,
        add_task_embedding=False,
        add_task_prompt_token_in_model=False,
        dataset="rico",
        discrete_x_grid=16,
        discrete_y_grid=16,
        lr=0.001,
        warmup_num_steps=1,
        out_dir=str(tmp_path),
        max_num_elements=5,
        eval_interval=1,
        enable_task_measure=False,
        decode_max_length=10,
        topk=5,
        temperature=0.7,
        sort_by_dict_label=False,
    )

    main.train(train_args)
    assert DummyTrainer.last_instance.called is True

    infer_args = SimpleNamespace(
        tasks="gen_ts,gen_r,gen_t",
        eval_tasks=None,
        eval_ckpt_tag="epoch_0",
        eval_seed=123,
        add_sep_token=False,
        add_task_prompt=False,
        add_task_prompt_token_in_model=False,
        dataset="rico",
        discrete_x_grid=16,
        discrete_y_grid=16,
        out_dir=str(tmp_path),
        sort_by_dict=False,
        load_vocab=True,
        max_num_elements=5,
    )

    main.inference(infer_args)
    assert DummyGenerator.last_instance.calls


def test_main_inference_uses_gen_t_constraint(tmp_path, monkeypatch):
    _install_deepspeed_stub()

    main = importlib.import_module("layoutformer_pp.main")
    main = importlib.reload(main)

    class DummyTokenizer:
        bos_token_id = 0
        pad_token_id = 0
        eos_token_id = 1

        def __len__(self):
            return 10

        def from_vocab(self, path):
            return None

    class DummyDataset:
        def __init__(self, tasks):
            self.tasks = tasks
            self.seq_processor = SimpleNamespace(index2label={1: "label_1"})
            self.colors = [(0, 0, 0)]

        def __len__(self):
            return 1

        def switch_task(self, task):
            self._task = task

    class DummyGenerator:
        last_instance = None

        def __init__(self, *args, **kwargs):
            DummyGenerator.last_instance = self
            self.last_constraint_fn = None

        def switch_task(self, task, saved_layouts=None):
            return None

        def __call__(self, inference_fn, draw_colors=None, constraint_fn=None):
            self.last_constraint_fn = constraint_fn

        def clean_up(self):
            return None

    class DummyConstraint:
        def __init__(self, *args, **kwargs):
            return None

    dummy_tokenizer = DummyTokenizer()

    monkeypatch.setattr(
        main, "create_tokenizer", lambda *args, **kwargs: dummy_tokenizer
    )
    monkeypatch.setattr(
        main, "build_model", lambda *args, **kwargs: torch.nn.Linear(1, 1)
    )
    monkeypatch.setattr(
        main,
        "create_dataset",
        lambda args, **kwargs: DummyDataset(args.tasks.split(",")),
    )
    monkeypatch.setattr(main, "create_fid_model", lambda *args, **kwargs: object())
    monkeypatch.setattr(main, "Generator", DummyGenerator)
    monkeypatch.setattr(
        main.constrained_decoding,
        "TransformerSortByDictLabelConstraint",
        DummyConstraint,
    )
    monkeypatch.setattr(
        main.constrained_decoding,
        "TransformerSortByDictLabelSizeConstraint",
        DummyConstraint,
    )
    monkeypatch.setattr(
        main.constrained_decoding,
        "TransformerSortByDictRelationConstraint",
        DummyConstraint,
    )

    infer_args = SimpleNamespace(
        tasks="gen_t",
        eval_tasks=None,
        eval_ckpt_tag="epoch_0",
        eval_seed=123,
        add_sep_token=False,
        add_task_prompt=False,
        add_task_prompt_token_in_model=False,
        dataset="rico",
        discrete_x_grid=16,
        discrete_y_grid=16,
        out_dir=str(tmp_path),
        sort_by_dict=False,
        load_vocab=True,
        max_num_elements=5,
    )

    main.inference(infer_args)
    assert isinstance(DummyGenerator.last_instance.last_constraint_fn, DummyConstraint)
