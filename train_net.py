from mmsf.utils.gpu import to_gpu
import torch
import numpy as np
from loguru import logger
from fvcore.common.config import CfgNode
from fvcore.nn.precise_bn import get_bn_modules, update_bn_stats
from tqdm import tqdm
import wandb

from mmsf.model.mmsf import MultimodalSlowFast
from mmsf.utils import metrics
from mmsf.utils.meters import EPICTrainMeter, EPICValMeter
import mmsf.utils.optimizer as optim
import mmsf.model.asf.checkpoint as asf_checkpoint
import mmsf.model.vsf.checkpoint as vsf_checkpoint
import mmsf.utils.checkpoint as cu
import mmsf.utils.distributed as du
import mmsf.datasets.loader as loader
import mmsf.utils.misc as misc
import mmsf.model.losses as losses
from eval_net import eval_epoch
from mmsf.datasets.mm_epickitchens import MultimodalEpicKitchens


def train_epoch(
    train_loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_meter: None,
    cur_epoch: int,
    cfg: CfgNode,
) -> None:
    # Enable train mode.
    model.train()

    if cfg.BN.FREEZE:
        model.module.freeze_fn("bn_statistics") if cfg.NUM_GPUS > 1 else model.freeze_fn("bn_statistics")

    train_meter.iter_tic()
    data_size = len(train_loader)

    for cur_iter, batch in enumerate(
        tqdm(
            train_loader,
            total=len(train_loader),
            desc="Training",
            unit="batch",
        ),
    ):
        specs, frames, labels, _, meta = batch

        if cfg.NUM_GPUS > 0:
            specs = to_gpu(specs)
            frames = to_gpu(frames)
            labels = to_gpu(labels)

        # Update the learning rate.
        lr = optim.get_epoch_lr(cur_epoch + float(cur_iter) / data_size, cfg)
        optim.set_lr(optimizer, lr)

        # Perform the forward pass.
        preds = model((specs, frames))
        verb_preds, noun_preds = preds

        # Explicitly declare reduction to mean.
        loss_fun = losses.get_loss_func(cfg.MODEL.LOSS_FUNC)(reduction="mean")

        # Compute the loss.
        loss_verb = loss_fun(verb_preds, labels["verb"])
        loss_noun = loss_fun(noun_preds, labels["noun"])
        loss = 0.5 * (loss_verb + loss_noun)

        # check Nan Loss.
        misc.check_nan_losses(loss)

        # Perform the backward pass.
        optimizer.zero_grad()
        loss.backward()
        # Update the parameters.
        optimizer.step()

        # Compute the verb accuracies.
        verb_top1_acc, verb_top5_acc = metrics.topk_accuracies(verb_preds, labels["verb"], (1, 5))

        # Gather all the predictions across all the devices.
        if cfg.NUM_GPUS > 1:
            loss_verb, verb_top1_acc, verb_top5_acc = du.all_reduce([loss_verb, verb_top1_acc, verb_top5_acc])

        # Copy the stats from GPU to CPU (sync point).
        loss_verb, verb_top1_acc, verb_top5_acc = (
            loss_verb.item(),
            verb_top1_acc.item(),
            verb_top5_acc.item(),
        )

        # Compute the noun accuracies.
        noun_top1_acc, noun_top5_acc = metrics.topk_accuracies(noun_preds, labels["noun"], (1, 5))

        # Gather all the predictions across all the devices.
        if cfg.NUM_GPUS > 1:
            loss_noun, noun_top1_acc, noun_top5_acc = du.all_reduce([loss_noun, noun_top1_acc, noun_top5_acc])

        # Copy the stats from GPU to CPU (sync point).
        loss_noun, noun_top1_acc, noun_top5_acc = (
            loss_noun.item(),
            noun_top1_acc.item(),
            noun_top5_acc.item(),
        )

        # Compute the action accuracies.
        action_top1_acc, action_top5_acc = metrics.multitask_topk_accuracies(
            (verb_preds, noun_preds), (labels["verb"], labels["noun"]), (1, 5)
        )
        # Gather all the predictions across all the devices.
        if cfg.NUM_GPUS > 1:
            loss, action_top1_acc, action_top5_acc = du.all_reduce([loss, action_top1_acc, action_top5_acc])

        # Copy the stats from GPU to CPU (sync point).
        loss, action_top1_acc, action_top5_acc = (
            loss.item(),
            action_top1_acc.item(),
            action_top5_acc.item(),
        )

        train_meter.iter_toc()

        # Update and log stats.
        train_meter.update_stats(
            (verb_top1_acc, noun_top1_acc, action_top1_acc),
            (verb_top5_acc, noun_top5_acc, action_top5_acc),
            (loss_verb, loss_noun, loss),
            lr,
            specs[0].size(0) * max(cfg.NUM_GPUS, 1),
        )

        train_meter.log_iter_stats(cur_epoch, cur_iter)
        train_meter.iter_tic()

    # Log epoch stats.
    train_meter.log_epoch_stats(cur_epoch)
    train_meter.reset()


def train_model(cfg: CfgNode) -> None:
    logger.warning("Training model.")

    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    logger.info(f"Training with config:\n{cfg}")

    model = MultimodalSlowFast(cfg=cfg)

    if cfg.NUM_GPUS > 0:
        cur_device = torch.cuda.current_device()
        model = model.cuda(device=cur_device)

    # Freeze the weights of the audio_model and video_model
    if cfg.MODEL.FREEZE_MOD_SPECIFIC_WEIGHTS:
        freeze_mod_specific_weights(model)

    misc.log_model_info(model)

    if cfg.BN.FREEZE:
        model.audio_model.freeze_fn("bn_parameters")
        model.video_model.freeze_fn("bn_parameters")

    optimizer = optim.construct_optimizer(model, cfg=cfg)

    # TODO: Add the case when the training resumes from a checkpoint
    load_weights(cfg=cfg, model=model)
    start_epoch = 0

    train_loader = loader.construct_loader(cfg=cfg, split="train", dataset_class=MultimodalEpicKitchens)
    val_loader = loader.construct_loader(cfg=cfg, split="val", dataset_class=MultimodalEpicKitchens)

    logger.info(f"Train Loader: {len(train_loader):,} batches of size {cfg.TRAIN.BATCH_SIZE}")
    logger.info(f"Val Loader: {len(val_loader):,} batches of size {cfg.TRAIN.BATCH_SIZE}")

    if cfg.WANDB.ENABLE:
        project_name = "MMMid-SlowFast"

        if not cfg.MODEL.FREEZE_MOD_SPECIFIC_WEIGHTS:
            project_name += "-full"

        project_name += f"-lr={cfg.SOLVER.BASE_LR}"
        project_name += f"-{cfg.SOLVER.OPTIMIZING_METHOD}"

        wandb.init(project=project_name, config=cfg)
        wandb.watch(model)

    # TODO: Create meters
    train_meter = EPICTrainMeter(len(train_loader), cfg)
    val_meter = EPICValMeter(len(val_loader), cfg)

    logger.info("Start training.")

    # Print the number of trainable parameters
    logger.warning(f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    for cur_epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCH):
        logger.success(f"Starting epoch {cur_epoch}...")
        loader.shuffle_dataset(loader=train_loader, cur_epoch=cur_epoch)

        train_epoch(
            train_loader=train_loader,
            model=model,
            optimizer=optimizer,
            train_meter=train_meter,
            cur_epoch=cur_epoch,
            cfg=cfg,
        )

        # Compute precise BN stats.
        if cfg.BN.USE_PRECISE_STATS and len(get_bn_modules(model)) > 0:
            calculate_and_update_precise_bn(train_loader, model, cfg)

        # Save a checkpoint.
        if cu.is_checkpoint_epoch(cfg, cur_epoch):
            cu.save_checkpoint(cfg.OUTPUT_DIR, model, optimizer, cur_epoch, cfg)

        # Evaluate the model on validation set.
        if misc.is_eval_epoch(cfg, cur_epoch):
            is_best_epoch = eval_epoch(val_loader, model, val_meter, cur_epoch, cfg)

            if is_best_epoch:
                cu.save_checkpoint(cfg.OUTPUT_DIR, model, optimizer, cur_epoch, cfg, is_best_epoch=is_best_epoch)


def load_weights(
    cfg: CfgNode,
    model: torch.nn.Module,
) -> None:
    if cfg.TRAIN.CHECKPOINT_FILE_PATH_ASF:
        asf_checkpoint.load_train_checkpoint(
            cfg=cfg,
            model=model.audio_model,
            optimizer=None,
        )

    if cfg.TRAIN.CHECKPOINT_FILE_PATH_VSF:
        _ = vsf_checkpoint.load_checkpoint(
            path_to_checkpoint=cfg.TRAIN.CHECKPOINT_FILE_PATH_VSF,
            data_parallel=cfg.NUM_GPUS > 1,
            model=model.video_model,
            optimizer=None,
            inflation=cfg.TRAIN.CHECKPOINT_INFLATE_VSF,
            convert_from_caffe2=False,
        )


def calculate_and_update_precise_bn(loader, model, cfg):
    """
    Update the stats in bn layers by calculate the precise stats.
    Args:
        loader (loader): data loader to provide training data.
        model (model): model to update the bn stats.
        num_iters (int): number of iterations to compute and update the bn stats.
    """

    def _gen_loader():
        for batch in loader:
            specs, frames, labels, _, meta = batch
            if cfg.NUM_GPUS > 0:
                specs = to_gpu(specs)
                frames = to_gpu(frames)
                labels = to_gpu(labels)

            yield (specs, frames)

    # Update the bn stats.
    update_bn_stats(model, _gen_loader(), cfg.BN.NUM_BATCHES_PRECISE)


def freeze_mod_specific_weights(model: torch.nn.Module) -> None:
    logger.info("Freezing audio_model and video_model weights.")
    assert hasattr(model, "audio_model") and hasattr(
        model, "video_model"
    ), f"Model {model.__class__.__name__} does not have audio_model or video_model attributes."

    # Freeze audio_model and video_model parameters
    for param in model.audio_model.parameters():
        param.requires_grad = False
    for param in model.video_model.parameters():
        param.requires_grad = False
