import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import chain
import warnings

from encoding_custom.utils.swiftnet_utils import _BNReluConv, upsample

from encoding_custom.models.base import BaseNet
from encoding_custom.datasets import datasets


class SwiftNet(nn.Module):
    def __init__(
        self,
        num_classes,
        num_inst_classes=None,
        use_bn=True,
        k=1,
        bias=True,
        loss_ret_additional=False,
        upsample_logits=True,
        logit_class=_BNReluConv,
        multiscale_factors=(0.5, 0.75, 1.5, 2.0),
    ):
        super(SwiftNet, self).__init__()
        self.num_classes = num_classes
        self.logits = logit_class(
            128, self.num_classes, batch_norm=use_bn, k=k, bias=bias
        )
        self.upsample_logits = upsample_logits
        self.multiscale_factors = multiscale_factors

    def forward(self, features):
        logits = self.logits.forward(features)
        return logits


class SwiftNet_wrapper(BaseNet):
    def __init__(self, num_classes, backbone, aux=False, se_loss=False, **kwargs):
        super().__init__(num_classes, backbone, aux, se_loss, **kwargs)
        self.head = SwiftNet(num_classes)

    def forward(self, img):
        image_size = img.size()[-2:]
        features, _ = self.base_forward(img)
        logits = self.head(features)
        if (not self.training) or self.head.upsample_logits:
            logits = upsample(logits, image_size)
        return [logits]


def get_swiftnet(
    dataset="", backbone="", pretrained=False, root="~/.encoding/models", **kwargs
):
    return SwiftNet_wrapper(
        datasets[dataset.lower()].NUM_CLASS, backbone, pretrained=pretrained, **kwargs,
    )
