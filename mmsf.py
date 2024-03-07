import torch.nn as nn
import torch
import config

from audio_slowfast.models.build import build_model as build_audio_model
from slowfast.models.build import build_model as build_video_model

from fvcore.common.config import CfgNode


class MultimodalSlowFast(nn.Module):
    def __init__(self, cfg: CfgNode) -> None:
        self.audio_model = build_audio_model(cfg)
        self.video_model = build_video_model(cfg)

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
