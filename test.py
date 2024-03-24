import torch
from loguru import logger
from fvcore.common.config import CfgNode


def test_epoch() -> None:
    pass


def test_model(model: torch.nn.Module, audio_cfg: CfgNode, video_cfg: CfgNode) -> None:
    logger.warning("Testing model.")
    pass
