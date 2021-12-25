import torch
import math


def optimizer_builder(model, cfg):
    if cfg.SOLVER.OPTIMIZING_METHOD == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=cfg.SOLVER.BASE_LR,
            momentum=cfg.SOLVER.MOMENTUM,
            weight_decay=float(cfg.SOLVER.WEIGHT_DECAY),
        )
    elif cfg.SOLVER.OPTIMIZING_METHOD == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=cfg.SOLVER.BASE_LR,
            betas=(0.9, 0.999),
            weight_decay=float(cfg.SOLVER.WEIGHT_DECAY),
        )
    else:
        raise NotImplementedError(
            "Not support {} optimizer".format(cfg.SOLVER.OPTIMIZING_METHOD)
        )


def get_cur_lr(cur_epoch, cfg):

    # Lear WarmUp
    if cur_epoch < cfg.SOLVER.WARMUP_EPOCH:
        return (
            cfg.SOLVER.WARMUP_START_LR
            + (cfg.SOLVER.BASE_LR-cfg.SOLVER.WARMUP_START_LR) 
            * cur_epoch / cfg.SOLVER.WARMUP_EPOCH
        )
    else:
        # Cosine scheduler
        if cfg.SOLVER.LR_POLICY == 'cosine':
            return (
                cfg.SOLVER.BASE_LR
                * (math.cos(math.pi * (cur_epoch-cfg.SOLVER.WARMUP_EPOCH) 
                / (cfg.SOLVER.MAX_EPOCH-cfg.SOLVER.WARMUP_EPOCH)) + 1.0)
                * 0.5
            )
        # Multi-step scheduler
        elif cfg.SOLVER.LR_POLICY == 'multistep':
            multi_step= [cfg.SOLVER.WARMUP_EPOCH-1]+ cfg.SOLVER.MULTI_STEP + [cfg.SOLVER.MAX_EPOCH+1]
            for i in range(len(multi_step)):
                if multi_step[i] >= cur_epoch:
                    return cfg.SOLVER.BASE_LR * pow(0.1,i-1)


def set_lr(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr
