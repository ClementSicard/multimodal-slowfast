# Add 'asf' folder to the path
import sys

sys.path.append("asf")
sys.path.append("vsf")

from asf.audio_slowfast.config.defaults import get_cfg as get_audio_cfg  # noqa: E402
from asf.audio_slowfast.models.build import build_model as build_audio_model  # noqa: E402
from asf.audio_slowfast.datasets import build_dataset as build_audio_dataset  # noqa: E402
from asf.audio_slowfast.utils import checkpoint as asf_cu  # noqa: E402

from vsf.slowfast.config.defaults import get_cfg as get_video_cfg  # noqa: E402
from vsf.slowfast.models.build import build_model as build_video_model  # noqa: E402
from vsf.slowfast.datasets import build_dataset as build_video_dataset  # noqa: E402
from vsf.slowfast.utils import checkpoint as vsf_cu  # noqa: E402
from typing import Dict, Any, Literal  # noqa: E402


def load_audio_config(args):
    return _load_config(args, type_="audio")


def load_video_config(args):
    return _load_config(args, type_="video")


def _load_config(args: Dict[str, Any], type_: Literal["audio", "video"] = "audio"):
    """
    Given the arguemnts, load and initialize the AUDIO config.
    Args:
        args (argument): arguments includes `shard_id`, `num_shards`,
            `init_method`, `cfg_file`, and `opts`.
    """

    # Setup cfg.
    cfg = get_audio_cfg() if type_ == "audio" else get_video_cfg()
    # Load config from cfg.
    key = "audio_cfg" if type_ == "audio" else "video_cfg"

    if args.get(key) is not None:
        cfg.merge_from_file(args.get(key))

    # Load config from command line, overwrite config from opts.
    if args.get("opts") is not None:
        cfg.merge_from_list(args.get("opts"))

    # Inherit parameters from args.
    if hasattr(args, "num_shards") and hasattr(args, "shard_id"):
        cfg.NUM_SHARDS = args.num_shards
        cfg.SHARD_ID = args.shard_id
    if hasattr(args, "rng_seed"):
        cfg.RNG_SEED = args.rng_seed
    if hasattr(args, "output_dir"):
        cfg.OUTPUT_DIR = args.output_dir

    return cfg
