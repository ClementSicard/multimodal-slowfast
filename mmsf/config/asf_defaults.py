#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Configs."""
from fvcore.common.config import CfgNode

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C = CfgNode()

# -----------------------------------------------------------------------------
# ResNet options
# -----------------------------------------------------------------------------
_C.RESNET = CfgNode()

# Transformation function.
_C.RESNET.TRANS_FUNC = "bottleneck_transform"

# Number of groups. 1 for ResNet, and larger than 1 for ResNeXt).
_C.RESNET.NUM_GROUPS = 1

# Width of each group (64 -> ResNet; 4 -> ResNeXt).
_C.RESNET.WIDTH_PER_GROUP = 64

# Apply relu in a inplace manner.
_C.RESNET.INPLACE_RELU = True

# Apply stride to 1x1 conv.
_C.RESNET.STRIDE_1X1 = False

#  If true, initialize the gamma of the final BN of each block to zero.
_C.RESNET.ZERO_INIT_FINAL_BN = False

# Number of weight layers.
_C.RESNET.DEPTH = 50

# If the current block has more than NUM_BLOCK_TEMP_KERNEL blocks, use temporal
# kernel of 1 for the rest of the blocks.
_C.RESNET.NUM_BLOCK_TEMP_KERNEL = [[3], [4], [6], [3]]

# Size of stride on different res stages.
_C.RESNET.FREQUENCY_STRIDES = [[1], [2], [2], [2]]

# Size of dilation on different res stages.
_C.RESNET.FREQUENCY_DILATIONS = [[1], [1], [1], [1]]


# -----------------------------------------------------------------------------
# Model options
# -----------------------------------------------------------------------------
_C.MODEL = CfgNode()

# Model architecture.
_C.MODEL.ARCH = "slowfast"

_C.MODEL.CLIP_MODEL = "ViT-B/32"

# Model name
_C.MODEL.MODEL_NAME = "SlowFast"

# The number of classes to predict for the model.
_C.MODEL.NUM_CLASSES = [400]


_C.MODEL.GRU_HIDDEN_SIZE = 512
_C.MODEL.GRU_NUM_LAYERS = 2

# The vocab files.
_C.MODEL.VOCAB_FILE = ""

_C.MODEL.ONLY_ACTION_RECOGNITION = False

# Loss function.
_C.MODEL.LOSS_FUNC = "cross_entropy"
_C.MODEL.STATE_LOSS_FUNC = "masked_loss"

# Model architectures that has one single pathway.
_C.MODEL.SINGLE_PATHWAY_ARCH = ["slow", "fast"]

# Model architectures that has multiple pathways.
_C.MODEL.MULTI_PATHWAY_ARCH = ["slowfast"]

# Dropout rate before final projection in the backbone.
_C.MODEL.DROPOUT_RATE = 0.5

# Randomly drop rate for Res-blocks, linearly increase from res2 to res5
_C.MODEL.DROPCONNECT_RATE = 0.0

# The std to initialize the fc layer(s).
_C.MODEL.FC_INIT_STD = 0.01

# Activation layer for the output head.
_C.MODEL.HEAD_ACT = "softmax"
_C.MODEL.PDDL_ATTRIBUTES = "softmax"


# -----------------------------------------------------------------------------
# SlowFast options
# -----------------------------------------------------------------------------
_C.SLOWFAST = CfgNode()

# Corresponds to the inverse of the channel reduction ratio, $\beta$ between
# the Slow and Fast pathways.
_C.SLOWFAST.BETA_INV = 8

# Corresponds to the frame rate reduction ratio, $\alpha$ between the Slow and
# Fast pathways.
_C.SLOWFAST.ALPHA = 8

# Ratio of channel dimensions between the Slow and Fast pathways.
_C.SLOWFAST.FUSION_CONV_CHANNEL_RATIO = 2

# Kernel dimension used for fusing information from Fast pathway to Slow
# pathway.
_C.SLOWFAST.FUSION_KERNEL_SZ = 5


# -----------------------------------------------------------------------------
# Audio data options
# -----------------------------------------------------------------------------
_C.AUDIO_DATA = CfgNode()

# Sampling rate of audio (in kHz)
_C.AUDIO_DATA.SAMPLING_RATE = 24000

_C.AUDIO_DATA.N_FFT = 2048

# Duration of audio clip from which to extract the spectrogram
_C.AUDIO_DATA.CLIP_SECS = 1.279

_C.AUDIO_DATA.WINDOW_LENGTH = 10.0

_C.AUDIO_DATA.HOP_LENGTH = 5.0

# Number of timesteps of the input spectrogram
_C.AUDIO_DATA.NUM_FRAMES = 256

# Number of frequencies of the input spectrogram
_C.AUDIO_DATA.NUM_FREQUENCIES = 128

# Overlap duration of two consecutive spectrograms, in seconds.
_C.AUDIO_DATA.SPECTROGRAM_OVERLAP = 1.0

_C.AUDIO_DATA.MAX_NB_SPECTROGRAMS = 15


_C.AUGMENT = CfgNode()

_C.AUGMENT.BALANCE = True
_C.AUGMENT.ENABLE = False
_C.AUGMENT.FACTOR = 1.0


def _assert_and_infer_cfg(cfg):
    # RESNET assertions.
    assert cfg.RESNET.NUM_GROUPS > 0
    assert cfg.RESNET.WIDTH_PER_GROUP > 0
    assert cfg.RESNET.WIDTH_PER_GROUP % cfg.RESNET.NUM_GROUPS == 0

    return cfg


def get_cfg():
    """
    Get a copy of the default config.
    """
    return _assert_and_infer_cfg(_C.clone())
