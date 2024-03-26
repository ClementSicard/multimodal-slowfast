import torch.nn as nn
import torch
from mmsf.model.asf.asf import AudioSlowFast
from mmsf.model.vsf.vsf import SlowFast as VideoSlowFast

from fvcore.common.config import CfgNode
from loguru import logger

from mmsf.model.head import MMMidSlowFastHead


class MultimodalSlowFast(nn.Module):
    def __init__(self, cfg: CfgNode) -> None:
        super(MultimodalSlowFast, self).__init__()

        self.cfg = cfg

        logger.warning(f"Initializing {self.__class__.__name__} model")

        self.audio_model = AudioSlowFast(cfg=self.cfg)
        self.video_model = VideoSlowFast(cfg=self.cfg)

        self.head = MMMidSlowFastHead(
            dim_in=[2304, 2304],
            num_classes=cfg.MODEL.NUM_CLASSES,
            act_func=cfg.MODEL.ACTIVATION_FUNC,
        )

    def forward(self, batch) -> None:
        specs, frames = batch

        audio_features = self.audio_model(specs)
        video_features = self.video_model(frames)

        audio_features = audio_features.unsqueeze(1)

        x = self.head(audio_features, video_features)

        return x
