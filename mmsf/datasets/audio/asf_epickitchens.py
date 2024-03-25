import os
from typing import List, Type

import h5py
import pandas as pd
import torch
import torch.utils.data
from fvcore.common.config import CfgNode
from fvcore.common.file_io import PathManager
from loguru import logger

from mmsf.datasets.audio.augmentations.transforms import get_transforms
from mmsf.datasets.audio.augmentations.spec_augment import spec_augment
import mmsf.datasets.audio.utils as utils
from mmsf.datasets.audio.audio_loader_epic import pack_audio
from mmsf.datasets.audio.audio_record import AudioRecord
from mmsf.datasets.audio.epickitchens_record import EpicKitchensAudioRecord


class EpicKitchens(torch.utils.data.Dataset):
    def __init__(
        self,
        cfg: CfgNode,
        mode: str,
        record_type: Type[AudioRecord] = EpicKitchensAudioRecord,
        gru_format: bool = False,
        modes: List[str] = [
            "train",
            "val",
            "test",
            "train+val",
        ],
    ):
        assert mode in modes, "Split '{}' not supported for {}".format(mode, self.__class__.__name__)
        self.cfg = cfg
        self.mode = mode
        self.record_type = record_type
        self.gru_format = gru_format

        if self.mode in ["train", "val", "train+val"]:
            self._num_clips = 1
        elif self.mode in ["test"]:
            self._num_clips = cfg.TEST.NUM_ENSEMBLE_VIEWS if "GRU" not in cfg.TEST.DATASET else 1

        self.audio_dataset = h5py.File(self.cfg.EPICKITCHENS.AUDIO_DATA_FILE, "r")

        logger.info("Constructing {} dataset on split '{}'...".format(self.__class__.__name__, mode))
        self.unique_batch = cfg.EPICKITCHENS.SINGLE_BATCH
        if self.unique_batch:
            logger.warning("Using a SINGLE batch for debugging.")

        self.transforms = get_transforms()

        self._construct_loader()

    def _construct_loader(self):
        """
        Construct the audio loader.
        """
        if self.mode == "train":
            path_annotations_pickle = [
                os.path.join(
                    self.cfg.EPICKITCHENS.ANNOTATIONS_DIR,
                    self.cfg.EPICKITCHENS.TRAIN_LIST,
                )
            ]
        elif self.mode == "val":
            path_annotations_pickle = [
                os.path.join(
                    self.cfg.EPICKITCHENS.ANNOTATIONS_DIR,
                    self.cfg.EPICKITCHENS.VAL_LIST,
                )
            ]
        elif self.mode == "test":
            path_annotations_pickle = [
                os.path.join(
                    self.cfg.EPICKITCHENS.ANNOTATIONS_DIR,
                    self.cfg.EPICKITCHENS.TEST_LIST,
                )
            ]
        else:
            path_annotations_pickle = [
                os.path.join(self.cfg.EPICKITCHENS.ANNOTATIONS_DIR, file)
                for file in [
                    self.cfg.EPICKITCHENS.TRAIN_LIST,
                    self.cfg.EPICKITCHENS.VAL_LIST,
                ]
            ]

        for file in path_annotations_pickle:
            assert PathManager.exists(file), "{} dir not found".format(file)

        self._audio_records = []
        self._temporal_idx = []

        for file in path_annotations_pickle:
            file_df = pd.read_pickle(file)

            for tup in file_df.iterrows() if not self.unique_batch else file_df[: self.cfg.TRAIN.BATCH_SIZE].iterrows():
                for idx in range(self._num_clips):
                    self._audio_records.append(
                        self.record_type(tup, cfg=self.cfg),
                    )
                    self._temporal_idx.append(idx)
        assert len(self._audio_records) > 0, "Failed to load EPIC-KITCHENS split {} from {}".format(
            self.mode, path_annotations_pickle
        )
        logger.info(
            "Constructing {} dataloader (size: {:,}) from {}".format(
                self.__class__.__name__,
                len(self._audio_records),
                path_annotations_pickle,
            )
        )

    def __getitem__(self, index):
        """
        Given the audio index, return the spectrogram, label, audio
        index, and metadata.
        Args:
            index (int): the audio index provided by the pytorch sampler.
        Returns:
            spectrogram (tensor): the spectrogram sampled from the audio. The dimension
                is `channel` x `num frames` x `num frequencies`.
            label (int): the label of the current audio.
            index (int): Return the index of the audio.
        """
        if self.audio_dataset is None:
            self.audio_dataset = h5py.File(self.cfg.EPICKITCHENS.AUDIO_DATA_FILE, "r")

        if self.gru_format:
            return self._get_item_gru(index)
        else:
            return self._get_item_regular(index)

    def _get_item_regular(self, index: int):
        if self.mode in ["train", "val", "train+val"]:
            # -1 indicates random sampling.
            temporal_sample_index = -1
        elif self.mode in ["test"]:
            temporal_sample_index = self._temporal_idx[index]
        else:
            raise NotImplementedError("Does not support {} mode".format(self.mode))

        transformation = self._audio_records[index].transformation

        spectrogram = pack_audio(
            cfg=self.cfg,
            audio_dataset=self.audio_dataset,
            audio_record=self._audio_records[index],
            temporal_sample_index=temporal_sample_index,
            transform=self.transforms[transformation] if transformation != "none" else None,
        )

        # Normalization.
        spectrogram = spectrogram.float()
        if self.mode in ["train", "train+val"]:
            # Data augmentation.
            # C T F -> C F T
            spectrogram = spectrogram.permute(0, 2, 1)
            # SpecAugment
            spectrogram = spec_augment(spectrogram)
            # C F T -> C T F -> (1, 400, 128)
            spectrogram = spectrogram.permute(0, 2, 1)

        label = self._audio_records[index].label
        spectrogram = utils.pack_pathway_output(self.cfg, spectrogram)
        metadata = self._audio_records[index].metadata

        # TODO: remove this hack
        # spectrogram = [spectrogram, spectrogram]

        return spectrogram, label, index, metadata

    def __len__(self):
        return len(self._audio_records)
