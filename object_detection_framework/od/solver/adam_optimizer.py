import torch
from od.solver import registry


@registry.SOLVERS.register("ADAM_optimizer")
def ADAM_optimizer(cfg, model, lr=None):
    lr = cfg.SOLVER.BASE_LR if lr is None else lr
    return torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=cfg.SOLVER.WEIGHT_DECAY
    )
