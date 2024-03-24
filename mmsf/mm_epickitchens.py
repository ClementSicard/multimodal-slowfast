import torch
from fvcore.common.config import CfgNode
from config import build_audio_dataset, build_video_dataset


class MultimodalEpicKitchens(torch.utils.data.Dataset):
    """
    Multimodal dataset for the EPIC-Kitchens dataset
    """

    def __init__(self, audio_cfg: CfgNode, video_cfg: CfgNode, split: str) -> None:
        self.audio_dataset = build_audio_dataset(
            dataset_name=audio_cfg.TRAIN.DATASET_NAME,
            cfg=audio_cfg,
            mode=split,
        )
        self.video_dataset = build_video_dataset(cfg=video_cfg, mode=split)

        assert len(self.audio_dataset) == len(self.video_dataset), (
            "The audio and video datasets must have the same length but "
            + f"got {len(self.audio_dataset)=} and {len(self.video_dataset)=}"
        )

    def __getitem__(self, index: int) -> torch.Tensor:
        audio_data_point = self.audio_dataset[index]
        video_data_point = self.video_dataset[index]
        return audio_data_point, video_data_point
