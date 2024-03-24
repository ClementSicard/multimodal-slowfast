from loguru import logger
from mmsf.mmsf import MultimodalSlowFast
from mmsf.config import load_audio_config, load_video_config
from argparse import ArgumentParser, Namespace
from typing import Dict, Any
import json
import torch

from test import test_model
from train import train_model


C_vid, T_vid, H, W = 3, 32, 224, 224
C_aud, T_aud, F, alpha = 3, 100, 128, 4


def main(args: Dict[str, Any]) -> None:
    audio_args = args.copy()
    video_args = args.copy()
    audio_args["cfg_file"] = args["audio_cfg"]
    video_args["cfg_file"] = args["video_cfg"]
    audio_cfg = load_audio_config(args=audio_args)
    video_cfg = load_video_config(args=video_args)

    if not torch.cuda.is_available():
        logger.warning("No GPU found. Running on CPU.")
        audio_cfg.NUM_GPUS = 0
        video_cfg.NUM_GPUS = 0

        audio_cfg.WANDB.ENABLE = False
        video_cfg.WANDB.ENABLE = False

        audio_cfg.DATA_LOADER.NUM_WORKERS = 4
        video_cfg.DATA_LOADER.NUM_WORKERS = 4

        audio_cfg.TRAIN.BATCH_SIZE = 2
        video_cfg.TRAIN.BATCH_SIZE = 2

        audio_cfg.TEST.BATCH_SIZE = 1
        video_cfg.TEST.BATCH_SIZE = 1

    assert audio_cfg.TRAIN.BATCH_SIZE == video_cfg.TRAIN.BATCH_SIZE, (
        f"Audio and video batch sizes should be the same but got {audio_cfg.TRAIN.BATCH_SIZE=}"
        + f" and {video_cfg.TRAIN.BATCH_SIZE=} respectively."
    )

    assert audio_cfg.TEST.BATCH_SIZE == video_cfg.TEST.BATCH_SIZE, (
        f"Audio and video batch sizes should be the same but got {audio_cfg.TEST.BATCH_SIZE=}"
        + f" and {video_cfg.TEST.BATCH_SIZE=} respectively."
    )

    model = MultimodalSlowFast(audio_cfg=audio_cfg, video_cfg=video_cfg)

    if args.get("train"):
        train_model(model=model, audio_cfg=audio_cfg, video_cfg=video_cfg)

    if args.get("test"):
        test_model(model=model, audio_cfg=audio_cfg, video_cfg=video_cfg)

    logger.success("Done! 🚢")


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--audio-cfg",
        type=str,
        default="configs/asf/asf-original.yaml",
    )
    parser.add_argument(
        "--video-cfg",
        type=str,
        default="configs/vsf/vsf-train.yaml",
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
