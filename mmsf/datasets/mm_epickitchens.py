import torch
from fvcore.common.config import CfgNode
from mmsf.datasets.audio.asf_epickitchens import EpicKitchens as AudioEpicKitchens
from mmsf.datasets.video.vsf_epickitchens import EpicKitchens as VideoEpicKitchens
from loguru import logger


class MultimodalEpicKitchens(torch.utils.data.Dataset):
    """
    Multimodal dataset for the EPIC-Kitchens dataset
    """

    def __init__(self, cfg: CfgNode, split: str) -> None:
        self.audio_dataset = AudioEpicKitchens(cfg=cfg, mode=split)
        self.video_dataset = VideoEpicKitchens(cfg=cfg, mode=split)

        assert len(self.audio_dataset) == len(self.video_dataset), (
            "The audio and video datasets must have the same length but "
            + f"got {len(self.audio_dataset)=} and {len(self.video_dataset)=}"
        )

    def __getitem__(self, index: int) -> torch.Tensor:
        audio_data_point = self.audio_dataset[index]
        video_data_point = self.video_dataset[index]

        spectrogram, audio_label, audio_index, audio_metadata = audio_data_point
        frames, video_label, video_index, video_metadata = video_data_point

        assert len(audio_label) == len(video_label), (
            "The audio and video labels must have the same length but "
            + f"got {len(audio_label)=} and {len(video_label)=}"
        )

        for a, b in zip(audio_label, video_label):
            assert a == b, f"Audio label {a} does not match video label {b}"

        assert audio_index == video_index, f"Audio index {audio_index} does not match video index {video_index}"

        return spectrogram, frames, audio_label, audio_index, audio_metadata

    def __len__(self) -> int:
        assert len(self.audio_dataset) == len(self.video_dataset), (
            "The audio and video datasets must have the same length but "
            + f"got {len(self.audio_dataset)=} and {len(self.video_dataset)=}"
        )
        return len(self.audio_dataset)
