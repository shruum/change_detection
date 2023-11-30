import torch
from od.solver import registry
from torch.optim.lr_scheduler import _LRScheduler


@registry.SCHEDULERS.register("CosineLR")
def CosineLR(cfg, optimizer, milestones):
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=1, eta_min=0, last_epoch=-1
    )
    for param_group in scheduler.optimizer.param_groups:
        param_group["lr"] = param_group["lr"] / cfg.SOLVER.BATCH_SIZE
        param_group["weight_decay"] = (
            param_group["weight_decay"] * cfg.SOLVER.BATCH_SIZE
        )
    return scheduler
