import torch
import numpy as np
from loguru import logger
from fvcore.common.config import CfgNode
from mmsf.model.mmsf import MultimodalSlowFast
import mmsf.utils.optimizer as optim


def train_epoch() -> None:
    pass


def train_model(cfg: CfgNode) -> None:
    logger.warning("Training model.")

    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    logger.info(f"Training with config:\n{cfg}")

    model = MultimodalSlowFast(cfg=cfg)
    logger.warning(model)

    if cfg.BN.FREEZE:
        model.audio_model.freeze_fn("bn_parameters")
        model.video_model.freeze_fn("bn_parameters")

    optimizer = optim.construct_optimizer(model, cfg=cfg)
