import torch
from loguru import logger
from fvcore.common.config import CfgNode


def train_epoch() -> None:
    pass


def train_model(model: torch.nn.Module, audio_cfg: CfgNode, video_cfg: CfgNode) -> None:
    logger.warning("Training model.")
    pass
