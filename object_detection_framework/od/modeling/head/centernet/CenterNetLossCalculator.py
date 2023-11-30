# ------------------------------------------------------------------------------
# Portions of this code are from
# CornerNet (https://github.com/princeton-vl/CornerNet)
# Copyright (c) 2019, NavInfo Europe
# NOTE:Implementation of CenterNet Head in Detection as a service Framework
# ------------------------------------------------------------------------------
import torch
from od.modeling.head.centernet.losses import (
    FocalLoss,
    RegL1Loss,
    RegLoss,
    NormRegL1Loss,
    RegWeightedL1Loss,
)
from od.modeling.head.centernet.utils import _sigmoid


class CtdetLoss(torch.nn.Module):
    def __init__(self, cfg):
        super(CtdetLoss, self).__init__()

        self.crit = FocalLoss()
        self.crit_reg = RegL1Loss()
        self.crit_wh = RegL1Loss()
        self.cfg = cfg
        self.hm_weight = self.cfg.MODEL.HEAD.LOSS_WEIGHTS["hm"]
        self.wh_weight = self.cfg.MODEL.HEAD.LOSS_WEIGHTS["wh"]
        self.off_weight = self.cfg.MODEL.HEAD.LOSS_WEIGHTS["reg"]

    def forward(self, outputs, batch):
        batch = batch["boxes"]
        for batch_key in batch.keys():
            batch[batch_key] = batch[batch_key].cuda()

        hm_loss, wh_loss, off_loss = 0, 0, 0
        output = outputs[0]
        output["hm"] = _sigmoid(output["hm"])

        hm_loss += self.crit(output["hm"], batch["hm"])
        wh_loss += self.crit_wh(
            output["wh"], batch["reg_mask"], batch["ind"], batch["wh"]
        )
        off_loss += self.crit_reg(
            output["reg"], batch["reg_mask"], batch["ind"], batch["reg"]
        )

        total_loss = (
            self.hm_weight * hm_loss
            + self.wh_weight * wh_loss
            + self.off_weight * off_loss
        )
        loss_stats = {
            "loss": total_loss,
            "hm_loss": hm_loss,
            "wh_loss": wh_loss,
            "off_loss": off_loss,
        }
        return loss_stats
