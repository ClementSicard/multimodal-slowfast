import os
import pickle
from loguru import logger
from fvcore.common.config import CfgNode
import numpy as np
import torch
from tqdm import tqdm
from mmsf.datasets import loader
from mmsf.model.mmsf import MultimodalSlowFast
from mmsf.utils import misc
from mmsf.utils.meters import EPICTestMeter
import mmsf.utils.distributed as du
import mmsf.utils.checkpoint as cu


def perform_test(
    test_loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    test_meter: EPICTestMeter,
    cfg: CfgNode,
):
    # Enable eval mode.
    model.eval()

    test_meter.iter_tic()

    for cur_iter, (inputs, labels, video_idx, meta) in tqdm(
        enumerate(test_loader),
        total=len(test_loader),
        desc="Testing",
        unit="batch",
    ):
        # Transfer the data to the current GPU device.
        if cfg.NUM_GPUS > 0:
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)

            # Transfer the data to the current GPU device.
            labels = {k: v.cuda() for k, v in labels.items()}
            video_idx = video_idx.cuda()

        # Perform the forward pass.
        preds = model(inputs)

        # Gather all the predictions across all the devices to perform ensemble.
        if cfg.NUM_GPUS > 1:
            verb_preds, verb_labels, video_idx = du.all_gather([preds[0], labels["verb"], video_idx])

            noun_preds, noun_labels, video_idx = du.all_gather([preds[1], labels["noun"], video_idx])
            meta = du.all_gather_unaligned(meta)
            metadata = {"narration_id": []}
            for i in range(len(meta)):
                metadata["narration_id"].extend(meta[i]["narration_id"])
        else:
            metadata = meta
            verb_preds, verb_labels, video_idx = preds[0], labels["verb"], video_idx
            noun_preds, noun_labels, video_idx = preds[1], labels["noun"], video_idx
        test_meter.iter_toc()
        # Update and log stats.
        test_meter.update_stats(
            (verb_preds.detach().cpu(), noun_preds.detach().cpu()),
            (verb_labels.detach().cpu(), noun_labels.detach().cpu()),
            metadata,
            video_idx.detach().cpu(),
        )
        test_meter.log_iter_stats(cur_iter)

        test_meter.iter_tic()

    # Log epoch stats and print the final testing results.
    preds, labels, confusion_matrices, metadata = test_meter.finalize_metrics()
    test_meter.reset()

    return preds, labels, confusion_matrices, metadata


def test(cfg: CfgNode) -> None:
    logger.warning("Testing model.")

    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    logger.info(f"Training with config:\n{cfg}")

    model = MultimodalSlowFast(cfg=cfg)
    misc.log_model_info(model=model)

    if cfg.TEST.CHECKPOINT_FILE_PATH != "":
        cu.load_checkpoint(
            cfg.TEST.CHECKPOINT_FILE_PATH,
            model,
            cfg.NUM_GPUS > 1,
            None,
            inflation=False,
            convert_from_caffe2=cfg.TEST.CHECKPOINT_TYPE == "caffe2",
        )
    elif cu.has_checkpoint(cfg.OUTPUT_DIR):
        last_checkpoint = cu.get_last_checkpoint(cfg.OUTPUT_DIR)
        cu.load_checkpoint(last_checkpoint, model, cfg.NUM_GPUS > 1)
    elif cfg.TRAIN.CHECKPOINT_FILE_PATH != "":
        # If no checkpoint found in TEST.CHECKPOINT_FILE_PATH or in the current
        # checkpoint folder, try to load checkpint from
        # TRAIN.CHECKPOINT_FILE_PATH and test it.
        cu.load_checkpoint(
            cfg.TRAIN.CHECKPOINT_FILE_PATH,
            model,
            cfg.NUM_GPUS > 1,
            None,
            inflation=False,
            convert_from_caffe2=False,
        )
    else:
        raise NotImplementedError("Unknown way to load checkpoint.")

    # Create video testing loaders.
    test_loader = loader.construct_loader(cfg, "test", dataset_class=MultimodalSlowFast)
    logger.info("Testing model for {:,} iterations".format(len(test_loader)))

    test_meter = EPICTestMeter(
        len(test_loader.dataset) // (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS),
        cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS,
        cfg.MODEL.NUM_CLASSES,
        len(test_loader),
    )

    # Perform multi-view test on the entire dataset.
    preds, labels, confusion_matrices, metadata = perform_test(test_loader, model, test_meter, cfg)

    if du.is_master_proc():
        v_labels = labels[0]
        n_labels = labels[1]
        results = {
            "verb_output": preds[0],
            "noun_output": preds[1],
            "verb_cm": confusion_matrices[0],
            "noun_cm": confusion_matrices[1],
            "narration_id": metadata,
            "verb_labels": v_labels,
            "noun_labels": n_labels,
        }
        scores_path = os.path.join(cfg.OUTPUT_DIR, "scores")
        if not os.path.exists(scores_path):
            os.makedirs(scores_path)
        file_path = os.path.join(scores_path, cfg.EPICKITCHENS.TEST_SPLIT + ".pkl")
        pickle.dump(results, open(file_path, "wb"))

    logger.success("Testing done. 🎉")
