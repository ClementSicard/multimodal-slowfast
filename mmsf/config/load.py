from typing import Dict, Any
from mmsf.config.defaults import get_cfg


def load_config(args: Dict[str, Any]):
    """
    Given the arguemnts, load and initialize the AUDIO config.
    Args:
        args (argument): arguments includes `shard_id`, `num_shards`,
            `init_method`, `cfg_file`, and `opts`.
    """
    # Setup cfg.
    cfg = get_cfg()
    cfg.merge_from_file(args["config"])

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
