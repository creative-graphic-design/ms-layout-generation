from layoutformer_pp.trainer.basic_trainer import Trainer
from layoutformer_pp.trainer.multitask_trainer import MultiTaskTrainer, DSMultiTaskTrainer
from layoutformer_pp.trainer.utils import CheckpointMeasurement
from layoutformer_pp.trainer.generator import Generator


def get_trainer(args) -> Trainer:
    if args.trainer == 'basic':
        return MultiTaskTrainer
    elif args.trainer == 'deepspeed':
        return DSMultiTaskTrainer
    raise NotImplementedError(f"No Trainer: {args.trainer}")
