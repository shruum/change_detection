import torch
from torch import nn
from torch.nn import functional as F


class BottleneckResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        expansion_factor,
        stride=1,
        norm_layer=nn.BatchNorm2d,
        activ_layer=nn.ReLU,
    ):
        super(BottleneckResidualBlock, self).__init__()
        tc = expansion_factor * in_channels
        self.is_skip = (stride == 1) and (in_channels == out_channels)

        self.pointwise_conv1 = nn.Conv2d(in_channels, tc, 1, stride=1, bias=False)
        self.bn1 = norm_layer(tc)
        self.relu1 = activ_layer(inplace=True)

        self.depthwise_conv2 = nn.Conv2d(
            tc, tc, 3, stride=stride, padding=1, groups=tc, bias=False
        )
        self.bn2 = norm_layer(tc)
        self.relu2 = activ_layer(inplace=True)

        self.pointwise_conv3 = nn.Conv2d(tc, out_channels, 1, stride=1, bias=False)
        self.bn3 = norm_layer(out_channels)

    def forward(self, tensor):
        if self.is_skip:
            skipcon = tensor
        tensor = self.pointwise_conv1(tensor)
        tensor = self.bn1(tensor)
        tensor = self.relu1(tensor)

        tensor = self.depthwise_conv2(tensor)
        tensor = self.bn2(tensor)
        tensor = self.relu2(tensor)

        tensor = self.pointwise_conv3(tensor)
        tensor = self.bn3(tensor)

        if self.is_skip:
            tensor += skipcon
        return tensor
