from loguru import logger
from mmsf.config.load import load_config
from argparse import ArgumentParser, Namespace
from typing import Dict, Any
import json
import torch

from test_net import test
from train_net import train_model


def main(args: Dict[str, Any]) -> None:
    cfg = load_config(args=args)

    if not torch.cuda.is_available():
        logger.warning("No GPU found. Running on CPU.")
        cfg.NUM_GPUS = 0

        cfg.WANDB.ENABLE = True

        cfg.DATA_LOADER.NUM_WORKERS = 4

        cfg.TRAIN.BATCH_SIZE = 2

        cfg.TEST.BATCH_SIZE = 2

    if args.get("train"):
        train_model(cfg=cfg)

    if args.get("test"):
        test(cfg=cfg)

    logger.success("Done! 🚢")


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="configs/train-config.yaml",
    )
    parser.add_argument(
        "--train",
        action="store_true",
    )
    parser.add_argument(
        "--test",
        action="store_true",
    )
    args = vars(parser.parse_args())

    logger.info(f"Args:\n{json.dumps(args, indent=4)}")

    return args


if __name__ == "__main__":
    args = parse_args()
    main(args=args)
