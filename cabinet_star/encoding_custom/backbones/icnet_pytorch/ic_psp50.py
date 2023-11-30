import math
import torch.nn as nn
import torch.nn.functional as F

from encoding_custom.nn.spatial_pyramid_pooling import SpatialPyramidPooling

__all__ = [
    "ICPSP50",
    "ic_psp50",
]


class ICPSP50(nn.Module):
    def __init__(
        self, norm_layer=nn.BatchNorm2d, activ_layer=nn.ReLU,
    ):
        super(ICPSP50, self).__init__()
        self.base_inchannels = [128, 256, 512, 1024]
        # p1
        self.convx3_1 = Convx3(
            strides=[2, 2, 2], norm_layer=norm_layer, activ_layer=activ_layer
        )

        # p2
        # downsample and then
        self.convx3_2 = Convx3(
            strides=[2, 1, 1], norm_layer=norm_layer, activ_layer=activ_layer
        )
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.res_b1 = ICResidualBlock(64, 128, 3)
        self.res_b2a = ICResidualBlock(128, 256, 1, strides=[2, 1, 1])

        # p3
        # downsample and then
        self.res_b2b = ICResidualBlock(256, 256, 3)
        self.res_b3 = ICResidualBlock(256, 512, 6)
        self.res_b4 = ICResidualBlock(512, 1024, 3)

        self.SPP = SpatialPyramidPooling(scales=[1, 2, 3, 6])

    def forward(self, x):
        p1 = self.convx3_1(x)

        p2 = F.interpolate(x, scale_factor=0.5, mode="bilinear")
        p2 = self.convx3_2(p2)
        p2 = self.maxpool(p2)
        p2 = self.res_b1(p2)
        p2 = self.res_b2a(p2)

        p3 = F.interpolate(p2, scale_factor=0.5, mode="bilinear")
        p3 = self.res_b2b(p3)
        p3 = self.res_b3(p3)
        p3 = self.res_b4(p3)

        p3 = self.SPP(p3)
        return p1, p2, p3


class Convx3(nn.Module):
    def __init__(
        self,
        strides=[1, 1, 1],
        channels=[3, 32, 32, 64],
        norm_layer=nn.BatchNorm2d,
        activ_layer=nn.ReLU,
    ):
        super(Convx3, self).__init__()
        self.conv = [None] * 3
        self.bn = [None] * 3
        self.relu = [None] * 3
        for i in range(3):
            self.conv[i] = nn.Conv2d(
                channels[i], channels[i + 1], 3, strides[i], 1, bias=False
            )
            self.bn[i] = norm_layer(channels[i + 1])
            self.relu[i] = activ_layer(inplace=True)
            # Add MSRA weights initialization
            nn.init.kaiming_normal_(self.conv[i].weight, a=0, nonlinearity="relu")
        self.conv = nn.ModuleList(self.conv)
        self.bn = nn.ModuleList(self.bn)
        self.relu = nn.ModuleList(self.relu)

    def forward(self, x):
        for i in range(3):
            x = self.conv[i](x)
            x = self.bn[i](x)
            x = self.relu[i](x)
        return x


class ICResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        repeat,
        strides=[1, 1, 1],
        norm_layer=nn.BatchNorm2d,
        activ_layer=nn.ReLU,
    ):
        super(ICResidualBlock, self).__init__()
        self.repeat = repeat
        self.res_unit = [None] * self.repeat

        for i in range(self.repeat):
            self.res_unit[i] = ICResidualUnit(
                in_channels if i == 0 else out_channels,
                out_channels,
                strides=strides,
                norm_layer=nn.BatchNorm2d,
                activ_layer=nn.ReLU,
            )

        self.res_unit = nn.ModuleList(self.res_unit)

    def forward(self, x):
        for i in range(self.repeat):
            x = self.res_unit[i](x)
        return x


class ICResidualUnit(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        strides=[1, 1, 1],
        norm_layer=nn.BatchNorm2d,
        activ_layer=nn.ReLU,
    ):
        super(ICResidualUnit, self).__init__()

        self.skip = in_channels != out_channels
        self.conv_1 = nn.Conv2d(
            in_channels, out_channels // 4, 1, strides[0], bias=False
        )
        self.bn_1 = norm_layer(out_channels // 4)
        self.relu_1 = activ_layer(inplace=True)

        self.conv_2 = nn.Conv2d(
            out_channels // 4, out_channels // 4, 3, strides[1], 1, bias=False
        )
        self.bn_2 = norm_layer(out_channels // 4)
        self.relu_2 = activ_layer(inplace=True)

        self.conv_3a = nn.Conv2d(
            out_channels // 4, out_channels, 1, strides[2], 0, bias=False
        )
        self.bn_3a = norm_layer(out_channels)

        if self.skip:
            self.conv_3b = nn.Conv2d(
                in_channels, out_channels, 1, strides[0], bias=False
            )
            self.bn_3b = norm_layer(out_channels)

        self.relu_3 = activ_layer(inplace=True)

        # Add MSRA weights initialization
        nn.init.kaiming_normal_(self.conv_1.weight, a=0, nonlinearity="relu")
        nn.init.kaiming_normal_(self.conv_2.weight, a=0, nonlinearity="relu")
        nn.init.kaiming_normal_(self.conv_3a.weight, a=0, nonlinearity="relu")
        if self.skip:
            nn.init.kaiming_normal_(self.conv_3b.weight, a=0, nonlinearity="relu")

    def forward(self, x):
        y = self.conv_1(x)
        y = self.bn_1(y)
        y = self.relu_1(y)
        y = self.conv_2(y)
        y = self.bn_2(y)
        y = self.relu_2(y)
        y = self.conv_3a(y)
        y = self.bn_3a(y)
        if self.skip:
            x = self.conv_3b(x)
            x = self.bn_3b(x)
        return self.relu_3(x + y)


def ic_psp50(pretrained=False, root="", **kwargs):
    model = ICPSP50()
    if pretrained:
        assert False, "Pretrained model not implemented"
    return model
