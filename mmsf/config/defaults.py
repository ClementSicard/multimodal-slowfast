from fvcore.common.config import CfgNode
from mmsf.config.asf_defaults import get_cfg as get_asf_cfg
from mmsf.config.vsf_defaults import get_cfg as get_vsf_cfg

_C = CfgNode()

"""
Batch norm configuration
"""
_C.BN = CfgNode()

_C.BN.FREEZE = False

_C.BN.USE_PRECISE_STATS = False

_C.BN.NUM_BATCHES_PRECISE = 200

_C.BN.WEIGHT_DECAY = 0.0

_C.BN.NORM_TYPE = "batchnorm"

_C.BN.NUM_SPLITS = 1

_C.BN.NUM_SYNC_DEVICES = 1

"""
Train configuration
"""
_C.TRAIN = CfgNode()

_C.TRAIN.ENABLE = True

_C.TRAIN.DATASET = "EpicKitchens"
_C.TRAIN.BATCH_SIZE = 64

_C.TRAIN.EVAL_PERIOD = 1
_C.TRAIN.CHECKPOINT_PERIOD = 1

# Resume training from the latest checkpoint in the output directory.
_C.TRAIN.AUTO_RESUME = True

# Path to the checkpoint to load the initial weight.
_C.TRAIN.CHECKPOINT_FILE_PATH_ASF = ""
_C.TRAIN.CHECKPOINT_FILE_PATH_VSF = ""
_C.TRAIN.CHECKPOINT_FILE_PATH = ""

_C.TRAIN.CHECKPOINT_EPOCH_RESET = False

_C.TRAIN.CHECKPOINT_CLEAR_NAME_PATTERN = ()

_C.TRAIN.CHECKPOINT_TYPE_VSF = "pytorch"

_C.TRAIN.CHECKPOINT_INFLATE_VSF = False

_C.TRAIN.FINETUNE_VSF = False

"""
Test configuration
"""
_C.TEST = CfgNode()

_C.TEST.ENABLE = True

_C.TEST.DATASET = "EpicKitchens"
_C.TEST.BATCH_SIZE = 64

_C.TEST.CHECKPOINT_FILE_PATH = ""

_C.TEST.NUM_ENSEMBLE_VIEWS_ASF = 10
_C.TEST.NUM_ENSEMBLE_VIEWS_VSF = 10

_C.TEST.NUM_SPATIAL_CROPS_VSF = 3
_C.TEST.CHECKPOINT_TYPE_VSF = "pytorch"


"""
W&B configuration
"""
_C.WANDB = CfgNode()

_C.WANDB.ENABLE = False
_C.WANDB.RUN_ID = ""


"""
Data configuration
"""
_C.DATA = CfgNode()

_C.DATA.PATH_TO_DATA_DIR_VSF = ""

_C.DATA.INPUT_CHANNEL_NUM_ASF = [1, 1]

_C.DATA.INPUT_CHANNEL_NUM_VSF = [3, 3]

_C.DATA.MULTI_LABEL = False

_C.DATA.PATH_PREFIX = ""

_C.DATA.CROP_SIZE = 224

_C.DATA.NUM_FRAMES = 8

_C.DATA.SAMPLING_RATE = 8

_C.DATA.MEAN = [0.45, 0.45, 0.45]
_C.DATA.STD = [0.225, 0.225, 0.225]

_C.DATA.TRAIN_JITTER_SCALES_VSF = [256, 320]

# Crop sizes
_C.DATA.TRAIN_CROP_SIZE_VSF = 224
_C.DATA.TEST_CROP_SIZE_VSF = 256


"""
Solver configuration
"""
_C.SOLVER = CfgNode()

# Base learning rate.
_C.SOLVER.BASE_LR = 0.1

# Learning rate policy (see utils/lr_policy.py for options and examples).
_C.SOLVER.LR_POLICY = "steps_with_relative_lrs"

# Exponential decay factor.
_C.SOLVER.GAMMA = 0.1

# Step size for 'exp' and 'cos' policies (in epochs).
_C.SOLVER.STEP_SIZE = 1

# Steps for 'steps_' policies (in epochs).
_C.SOLVER.STEPS = []

# Learning rates for 'steps_' policies.
_C.SOLVER.LRS = []

# Maximal number of epochs.
_C.SOLVER.MAX_EPOCH = 300

# Momentum.
_C.SOLVER.MOMENTUM = 0.9

# Momentum dampening.
_C.SOLVER.DAMPENING = 0.0

# Nesterov momentum.
_C.SOLVER.NESTEROV = True

# L2 regularization.
_C.SOLVER.WEIGHT_DECAY = 1e-4

# Start the warm up from SOLVER.BASE_LR * SOLVER.WARMUP_FACTOR.
_C.SOLVER.WARMUP_FACTOR = 0.1

# Gradually warm up the SOLVER.BASE_LR over this number of epochs.
_C.SOLVER.WARMUP_EPOCHS = 0

# The start learning rate of the warm up.
_C.SOLVER.WARMUP_START_LR = 0.01

# Optimization method.
_C.SOLVER.OPTIMIZING_METHOD = "sgd"


"""
Misc configuration
"""
_C.NUM_GPUS = 1

_C.NUM_SHARDS = 1

_C.SHARD_ID = 0

_C.OUTPUT_DIR = "results"

_C.RNG_SEED = 1

_C.LOG_PERIOD = 10

_C.DIST_BACKEND = "nccl"


"""
Dataloader configuration
"""
_C.DATA_LOADER = CfgNode()
_C.DATA_LOADER.NUM_WORKERS = 8
_C.DATA_LOADER.PIN_MEMORY = True
_C.DATA_LOADER.ENABLE_MULTI_THREAD_DECODE = False


"""
EPIC-KITCHENS Dataset options
"""
_C.EPICKITCHENS = CfgNode()

_C.EPICKITCHENS.VISUAL_DATA_DIR = ""

_C.EPICKITCHENS.AUDIO_DATA_FILE = ""

_C.EPICKITCHENS.ANNOTATIONS_DIR = ""

_C.EPICKITCHENS.SINGLE_BATCH = False

_C.EPICKITCHENS.TRAIN_LIST = "EPIC_100_train.pkl"

_C.EPICKITCHENS.VAL_LIST = "EPIC_100_validation.pkl"

_C.EPICKITCHENS.TEST_LIST = "EPIC_100_validation.pkl"

_C.EPICKITCHENS.TEST_SPLIT = "validation"

_C.EPICKITCHENS.TRAIN_PLUS_VAL = False

_C.EPICKITCHENS.VIDEO_DURS = "EPIC_100_video_info.csv"


def _assert_and_infer_cfg(cfg: CfgNode) -> CfgNode:
    # BN assertions.
    if cfg.BN.USE_PRECISE_STATS:
        assert cfg.BN.NUM_BATCHES_PRECISE >= 0
    # TRAIN assertions.
    assert cfg.TRAIN.CHECKPOINT_TYPE_VSF in ["pytorch", "caffe2"]
    assert cfg.TRAIN.BATCH_SIZE % cfg.NUM_GPUS == 0

    # TEST assertions.
    assert cfg.TEST.CHECKPOINT_TYPE_VSF in ["pytorch", "caffe2"]
    assert cfg.TEST.BATCH_SIZE % cfg.NUM_GPUS == 0
    assert cfg.TEST.NUM_SPATIAL_CROPS_VSF == 3

    # RESNET assertions.
    assert cfg.ASF.RESNET.NUM_GROUPS > 0
    assert cfg.ASF.RESNET.WIDTH_PER_GROUP > 0
    assert cfg.ASF.RESNET.WIDTH_PER_GROUP % cfg.ASF.RESNET.NUM_GROUPS == 0

    assert cfg.VSF.RESNET.NUM_GROUPS > 0
    assert cfg.VSF.RESNET.WIDTH_PER_GROUP > 0
    assert cfg.VSF.RESNET.WIDTH_PER_GROUP % cfg.VSF.RESNET.NUM_GROUPS == 0

    # General assertions.
    assert cfg.SHARD_ID < cfg.NUM_SHARDS
    return cfg


def get_cfg() -> CfgNode:
    """
    Get a copy of the default config.

    Returns
    -------
    `CfgNode`
        The default config.
    """
    cfg = _C.clone()

    # Get model-specific configs and create a node for each of them
    cfg.ASF = get_asf_cfg()
    cfg.VSF = get_vsf_cfg()
    return _assert_and_infer_cfg(cfg)
