import torch.nn as nn
import torch

from asf.audio_slowfast.models.build import build_model as build_audio_model
from vsf.slowfast.models.build import build_model as build_video_model


class MultimodalSlowFast(nn.Module):
    def __init__(self, cfg) -> None:
        self.audio_model = build_audio_model(cfg)
        self.video_model = build_video_model(cfg)
