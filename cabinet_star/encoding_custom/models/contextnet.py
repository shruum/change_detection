import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import functional

from encoding_custom.models.base import BaseNet
from encoding_custom.datasets import datasets
from encoding_custom.nn.spatial_pyramid_pooling import SpatialPyramidPooling

from .seg_opr.loss_opr import SigmoidFocalLoss, ProbOhemCrossEntropy2d
from .seg_opr.seg_oprs import ConvBnRelu, AttentionRefinement, FeatureFusion


class ContextNet(nn.Module):
    def __init__(
        self, n_classes, norm_layer=nn.BatchNorm2d, activ_layer=nn.ReLU,
    ):
        super(ContextNet, self).__init__()
        self.spatialpath = SpatialPath(norm_layer, activ_layer)
        self.fusion = Fusion(n_classes, norm_layer, activ_layer)

    def forward(self, data, context):
        detail = self.spatialpath(data)
        return self.fusion(context, detail)


class SpatialPath(nn.Module):
    def __init__(self, norm_layer=nn.BatchNorm2d, activ_layer=nn.ReLU):
        super(SpatialPath, self).__init__()
        self.conv_1 = nn.Conv2d(3, 32, 3, 2, 1, bias=False)
        self.bn_1 = norm_layer(32)
        self.relu_1 = activ_layer(inplace=True)
        self.dwconv_1 = DepthwiseConv2DBNReLU(
            32,
            64,
            3,
            stride=2,
            padding=1,
            bias=False,
            norm_layer=norm_layer,
            activ_layer=activ_layer,
        )
        self.dwconv_2 = DepthwiseConv2DBNReLU(
            64,
            128,
            3,
            stride=2,
            padding=1,
            bias=False,
            norm_layer=norm_layer,
            activ_layer=activ_layer,
        )
        self.dwconv_3 = DepthwiseConv2DBNReLU(
            128,
            128,
            3,
            stride=1,
            padding=1,
            bias=False,
            norm_layer=norm_layer,
            activ_layer=activ_layer,
        )

    def forward(self, x):
        x = self.conv_1(x)
        x = self.bn_1(x)
        x = self.relu_1(x)
        x = self.dwconv_1(x)
        x = self.dwconv_2(x)
        x = self.dwconv_3(x)
        return x


class Fusion(nn.Module):
    def __init__(self, n_classes, norm_layer=nn.BatchNorm2d, activ_layer=nn.ReLU):
        super(Fusion, self).__init__()
        self.conv1d = nn.Conv2d(128, 128, 1, bias=False)
        self.bn1d = norm_layer(128)

        self.conv1c = nn.Conv2d(128, 128, 1, bias=False)
        self.bn1c = norm_layer(128)
        self.relu1c = activ_layer(inplace=True)
        # ----------------------------
        self.drop2 = nn.Dropout()
        self.conv2 = nn.Conv2d(128, n_classes, 1, bias=True)
        # ----------------------------
        self.depthwise_conv3 = nn.Conv2d(
            128, 128, 3, stride=1, dilation=4, padding=4, groups=128, bias=False
        )
        self.bn3 = norm_layer(128)
        self.relu3 = activ_layer(inplace=True)
        self.conv4 = nn.Conv2d(128, 128, 1, bias=False)

        self.relu5 = activ_layer(inplace=True)
        self.drop6 = nn.Dropout()
        self.conv6 = nn.Conv2d(128, n_classes, 1, bias=True)

    def forward(self, context, detail):
        detail = self.conv1d(detail)
        detail = self.bn1d(detail)

        context = self.conv1c(context)
        context = self.bn1c(context)
        context = self.relu1c(context)
        context = F.interpolate(context, scale_factor=4, mode="bilinear")
        # -------------------------
        # Auxiliary out
        output_aux = self.drop2(context)
        output_aux = self.conv2(output_aux)
        # --------------------------
        context = self.depthwise_conv3(context)
        context = self.bn3(context)
        context = self.relu3(context)
        context = self.conv4(context)

        output = context + detail
        output = self.relu5(output)

        output = self.drop6(output)
        output = self.conv6(output)
        output = F.interpolate(output, scale_factor=8, mode="bilinear")
        return output, output_aux


class DepthwiseConv2DBNReLU(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        norm_layer=nn.BatchNorm2d,
        activ_layer=nn.ReLU,
        **kwargs
    ):
        super(DepthwiseConv2DBNReLU, self).__init__()
        kwargs["groups"] = in_channels
        self.dwconv = nn.Conv2d(in_channels, out_channels, kernel_size, **kwargs)
        self.pconv = nn.Conv2d(out_channels, out_channels, 1, bias=False)
        self.bn = norm_layer(out_channels)
        self.relu = activ_layer(inplace=True)

    def forward(self, tensor):
        tensor = self.dwconv(tensor)
        tensor = self.pconv(tensor)
        tensor = self.bn(tensor)
        tensor = self.relu(tensor)
        return tensor


class ContextNet_wrapper(BaseNet):
    def __init__(self, num_classes, backbone, aux=False, se_loss=False, **kwargs):
        super().__init__(num_classes, backbone, aux, se_loss, **kwargs)
        self.head = ContextNet(num_classes)
        self.aux_indexes = [1]
        self.num_outputs = 2

    def forward(self, img):
        context_path = self.base_forward(img)
        results = self.head(img, context_path)
        return results


def get_contextnet(
    dataset="", backbone="", pretrained=False, root="~/.encoding/models", **kwargs
):
    return ContextNet_wrapper(
        datasets[dataset.lower()].NUM_CLASS,
        backbone,
        pretrained=pretrained,
        root=root,
        **kwargs,
    )
