import torch
from torch import nn
from torch.nn import functional as F


class SpatialPyramidPooling(nn.Module):
    def __init__(self, scales=[1, 2, 3, 6], mode="avg"):
        super(SpatialPyramidPooling, self).__init__()
        if mode == "avg":
            self.pools = [nn.AdaptiveAvgPool2d(s) for s in scales]
        elif mode == "max":
            self.pools = [nn.AdaptiveMaxPool2d(s) for s in scales]

    def forward(self, in_tensor):
        out_tensor = [in_tensor]
        for pool in self.pools:
            out_tensor.append(
                F.interpolate(
                    pool(in_tensor),
                    size=(in_tensor.size()[2], in_tensor.size()[3]),
                    mode="bilinear",
                )
            )
        return sum(out_tensor)
