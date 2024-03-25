import torch
import numpy as np
from loguru import logger
from fvcore.common.config import CfgNode
from mmsf.model.mmsf import MultimodalSlowFast
import sf.slowfast.models.optimizer as optim
import sf.slowfast.utils.checkpoint as video_cu
import asf.audio_slowfast.utils.checkpoint as audio_cu


def train_epoch() -> None:
    pass


def train_model(model: torch.nn.Module, audio_cfg: CfgNode, video_cfg: CfgNode) -> None:
    logger.warning("Training model.")

    np.random.seed(audio_cfg.RNG_SEED)
    torch.manual_seed(audio_cfg.RNG_SEED)

    logger.info(f"Training with audio config:\n{audio_cfg}")
    logger.info(f"Training with video config:\n{video_cfg}")

    model = MultimodalSlowFast(audio_cfg=audio_cfg, video_cfg=video_cfg)
    logger.warning(model)

    if audio_cfg.BN.FREEZE:
        model.audio_model.freeze_fn("bn_parameters")

    if video_cfg.BN.FREEZE:
        model.video_model.freeze_fn("bn_parameters")

    optimizer = optim.construct_optimizer(model, video_cfg)
