import torch.nn as nn
import torch
from mmsf.model.asf.asf import AudioSlowFast
from mmsf.model.vsf.vsf import SlowFast as VideoSlowFast

from fvcore.common.config import CfgNode
from loguru import logger


class MultimodalSlowFast(nn.Module):
    def __init__(self, cfg: CfgNode) -> None:
        super(MultimodalSlowFast, self).__init__()

        self.cfg = cfg

        logger.warning(f"Initializing {self.__class__.__name__} model")

        self.audio_model = AudioSlowFast(cfg=self.cfg)
        self.video_model = VideoSlowFast(cfg=self.cfg)

        # Remove the last layer of the video model
        self.audio_model.head = nn.Identity()
        self.video_model.head = nn.Identity()

        self.fusion_fc = nn.Linear(2048, 2)

    def forward(self, x) -> None:
        audio, video = x

        audio_features = self.audio_model(audio)
        video_features = self.video_model(video)

        # Concatenate the features
        features = torch.cat((audio_features, video_features), dim=1)

        # Fusion
        output = self.fusion_fc(features)

        return output
