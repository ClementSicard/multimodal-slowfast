from mmsf.utils.gpu import to_gpu
import torch
from fvcore.common.config import CfgNode

from mmsf.utils import metrics
import mmsf.utils.distributed as du
from mmsf.utils.meters import EPICValMeter
from tqdm import tqdm


@torch.no_grad()
def eval_epoch(
    val_loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    val_meter: EPICValMeter,
    cur_epoch: int,
    cfg: CfgNode,
) -> bool:
    """
    Evaluate the model on the val set.

    Parameters
    ----------
    `val_loader` : `torch.utils.data.DataLoader`
        The validation data loader.

    `model` : `torch.nn.Module`
        The model to evaluate.

    `val_meter` : `EPICValMeter`
        The meter to store the validation stats.

    `cur_epoch` : `int`
        The current epoch.

    `cfg` : `CfgNode`
        The configurations.

    Returns
    -------
    `bool`
        Whether the current epoch is the best epoch.
    """
    # Evaluation mode enabled. The running stats would not be updated.
    model.eval()
    val_meter.iter_tic()

    for cur_iter, batch in enumerate(
        tqdm(
            val_loader,
            total=len(val_loader),
            desc="Validation",
            unit="batch",
        )
    ):
        specs, frames, labels, _, meta = batch

        if cfg.NUM_GPUS > 0:
            specs = to_gpu(specs)
            frames = to_gpu(frames)
            labels = to_gpu(labels)

        preds = model((specs, frames))

        verb_preds, noun_preds = preds
        # Compute the verb accuracies.
        verb_top1_acc, verb_top5_acc = metrics.topk_accuracies(verb_preds, labels["verb"], (1, 5))

        # Combine the errors across the GPUs.
        if cfg.NUM_GPUS > 1:
            verb_top1_acc, verb_top5_acc = du.all_reduce([verb_top1_acc, verb_top5_acc])

        # Copy the errors from GPU to CPU (sync point).
        verb_top1_acc, verb_top5_acc = verb_top1_acc.item(), verb_top5_acc.item()

        # Compute the noun accuracies.
        noun_top1_acc, noun_top5_acc = metrics.topk_accuracies(noun_preds, labels["noun"], (1, 5))

        # Combine the errors across the GPUs.
        if cfg.NUM_GPUS > 1:
            noun_top1_acc, noun_top5_acc = du.all_reduce([noun_top1_acc, noun_top5_acc])

        # Copy the errors from GPU to CPU (sync point).
        noun_top1_acc, noun_top5_acc = noun_top1_acc.item(), noun_top5_acc.item()

        # Compute the action accuracies.
        action_top1_acc, action_top5_acc = metrics.multitask_topk_accuracies(
            (verb_preds, noun_preds), (labels["verb"], labels["noun"]), (1, 5)
        )
        # Combine the errors across the GPUs.
        if cfg.NUM_GPUS > 1:
            action_top1_acc, action_top5_acc = du.all_reduce([action_top1_acc, action_top5_acc])

        # Copy the errors from GPU to CPU (sync point).
        action_top1_acc, action_top5_acc = action_top1_acc.item(), action_top5_acc.item()

        val_meter.iter_toc()

        # Update and log stats.
        val_meter.update_stats(
            (verb_top1_acc, noun_top1_acc, action_top1_acc),
            (verb_top5_acc, noun_top5_acc, action_top5_acc),
            specs[0].size(0) * max(cfg.NUM_GPUS, 1),
        )

        val_meter.log_iter_stats(cur_epoch, cur_iter)
        val_meter.iter_tic()

    # Log epoch stats.
    is_best_epoch = val_meter.log_epoch_stats(cur_epoch)
    val_meter.reset()
    return is_best_epoch
