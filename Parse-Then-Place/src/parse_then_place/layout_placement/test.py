import logging

from torch.utils.data import DataLoader

from parse_then_place.layout_placement.config.config import Config
from parse_then_place.layout_placement.dataset.dataset import PlacementDataset
from parse_then_place.layout_placement.placement_utils.utils import *
from parse_then_place.layout_placement.placement_utils.utils import Logger
from parse_then_place.layout_placement.trainer.trainer import Trainer

if __name__ == "__main__":
    logger = Logger()
    config = Config()

    logging.info("start testing placement model!!!!")
    set_seed(config.args.seed)

    test_set = PlacementDataset(config, split="prediction")
    test_loader = DataLoader(
        test_set,
        batch_size=config.args.batch_size,
        shuffle=False,
        num_workers=config.args.num_workers
    )
    logging.info(f"datasets loaded, test: {len(test_set)}")

    trainer = Trainer(config, device='cuda')
    trainer.test(test_loader)
