import math
import torch.nn as nn
import torch.nn.functional as F

from encoding_custom.nn.bottleneck_residual_block import BottleneckResidualBlock
from encoding_custom.nn.spatial_pyramid_pooling import SpatialPyramidPooling

__all__ = [
    "ContextPath",
    "contextpath",
]


class ContextPath(nn.Module):
    def __init__(
        self, norm_layer=nn.BatchNorm2d, activ_layer=nn.ReLU,
    ):
        super(ContextPath, self).__init__()
        self.conv_1 = nn.Conv2d(3, 32, 3, 2, 1, bias=False)
        self.bn_1 = norm_layer(32)
        self.relu_1 = activ_layer(inplace=True)
        self.b_res01 = BottleneckResidualBlock(32, 32, 1, 1, norm_layer, activ_layer)
        self.b_res02 = BottleneckResidualBlock(32, 32, 6, 1, norm_layer, activ_layer)
        self.b_res03 = BottleneckResidualBlock(32, 48, 6, 2, norm_layer, activ_layer)
        self.b_res04 = BottleneckResidualBlock(48, 48, 6, 1, norm_layer, activ_layer)
        self.b_res05 = BottleneckResidualBlock(48, 48, 6, 1, norm_layer, activ_layer)
        self.b_res06 = BottleneckResidualBlock(48, 64, 6, 2, norm_layer, activ_layer)
        self.b_res07 = BottleneckResidualBlock(64, 64, 6, 1, norm_layer, activ_layer)
        self.b_res08 = BottleneckResidualBlock(64, 64, 6, 1, norm_layer, activ_layer)
        self.b_res09 = BottleneckResidualBlock(64, 96, 6, 1, norm_layer, activ_layer)
        self.b_res10 = BottleneckResidualBlock(96, 96, 6, 1, norm_layer, activ_layer)
        self.b_res11 = BottleneckResidualBlock(96, 128, 6, 1, norm_layer, activ_layer)
        self.b_res12 = BottleneckResidualBlock(128, 128, 6, 1, norm_layer, activ_layer)

        self.relu_2 = activ_layer(inplace=True)
        self.SPP = SpatialPyramidPooling(scales=[1, 2, 4, 8])

    def forward(self, x):
        x = F.interpolate(x, scale_factor=0.25, mode="bilinear")
        x = self.conv_1(x)
        x = self.bn_1(x)
        x = self.relu_1(x)
        x = self.b_res01(x)
        x = self.b_res02(x)
        x = self.b_res03(x)
        x = self.b_res04(x)
        x = self.b_res05(x)
        x = self.b_res06(x)
        x = self.b_res07(x)
        x = self.b_res08(x)
        x = self.b_res09(x)
        x = self.b_res10(x)
        x = self.b_res11(x)
        x = self.b_res12(x)
        x = self.relu_2(x)
        x = self.SPP(x)
        return x


def contextpath(pretrained=False, root="", **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ContextPath()
    if pretrained:
        assert False, "Pretrained model not implemented"
        # model.load_state_dict(model_zoo.load_url(model_urls["resnet18"]), strict=False)
    return model
