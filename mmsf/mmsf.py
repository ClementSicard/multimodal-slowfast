import torch.nn as nn
import torch
from mmsf.config import build_audio_model, build_video_model

from fvcore.common.config import CfgNode
from loguru import logger


class MultimodalSlowFast(nn.Module):
    def __init__(self, audio_cfg: CfgNode, video_cfg: CfgNode) -> None:
        super(MultimodalSlowFast, self).__init__()

        logger.warning(f"{video_cfg.MODEL.MODEL_NAME=}")

        self.audio_model = build_audio_model(cfg=audio_cfg)
        self.video_model = build_video_model(cfg=video_cfg)

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
