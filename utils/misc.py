import utils.logger as _logger
import os
import numpy as np
import torch
import utils.distributed as dist

logger = _logger.get_logger(__name__)


def params_count(model):
    """
    Computes the number of parameters.
    Args:
        model (model): model to count the number of parameters.
    """
    return np.sum([p.numel() for p in model.parameters()]).item()


def log_model_info(model):
    """
    Logs info, includes number of parameters and gpu usage.
    Args:
        model (model): model to log the info.
    """
    logger.info("Model:\n{}".format(model))
    logger.info("Params: {:,}".format(params_count(model)))
    logger.info("nvidia-smi")
    os.system("nvidia-smi")


def is_checkpoint_epoch(cur_epoch, checkpoint_period):

    return (cur_epoch + 1) % checkpoint_period == 0


def is_eval_epoch(cur_epoch, eval_period, max_epoch):

    return (
        cur_epoch + 1
    ) % eval_period == 0 or cur_epoch + 1 == max_epoch


def save_checkpoint(path, model, optimizer, epoch, cfg):

    if not dist.is_master_proc(cfg.NUM_GPUS):
        return
    # Ensure that the checkpoint dir exists.
    os.makedirs(path, exist_ok=True)
    # Omit the DDP wrapper in the multi-gpu setting.
    sd = model.module.state_dict() if cfg.NUM_GPUS > 1 else model.state_dict()
    # Record the state.
    checkpoint = {
        "epoch": epoch,
        "model_state": sd,
        "optimizer_state": optimizer.state_dict(),
        "cfg": cfg.dump(),
    }
    # Write the checkpoint.
    torch.save(checkpoint, os.path.join(path,"checkpoint_epoch_{:05d}.pth.tar".format(epoch)))


def load_checkpoint(path, model, data_parallel=True, optimizer=None):
    assert os.path.exists(path), "checkpoint '{}' not found".format(path)

    ms = model.module if data_parallel else model

    state_dict = torch.load(path, map_location="cpu")

    ms.load_state_dict(state_dict["model_state"])

    if optimizer and "optimizer_state" in state_dict:
        optimizer.load_state_dict(state_dict["optimizer_state"])
        epoch = state_dict["epoch"]
    else:
        epoch = -1
    return epoch
