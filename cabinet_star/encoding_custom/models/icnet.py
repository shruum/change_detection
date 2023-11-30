import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import functional

from encoding_custom.models.base import BaseNet
from encoding_custom.datasets import datasets
from encoding_custom.nn.spatial_pyramid_pooling import SpatialPyramidPooling

from .seg_opr.loss_opr import SigmoidFocalLoss, ProbOhemCrossEntropy2d
from .seg_opr.seg_oprs import ConvBnRelu, AttentionRefinement, FeatureFusion


class ICNet(nn.Module):
    def __init__(
        self,
        out_planes,
        backbone_channels=[64, 128, 256, 512],
        norm_layer=nn.BatchNorm2d,
    ):
        super(ICNet, self).__init__()
        self.CFF1 = CascadeFeatureFusion(
            [backbone_channels[3], backbone_channels[1]],
            backbone_channels[0],
            n_classes=out_planes,
        )
        self.CFF2 = CascadeFeatureFusion(
            [backbone_channels[0], backbone_channels[0] // 2],
            backbone_channels[0],
            n_classes=out_planes,
        )
        self.conv_out = nn.Conv2d(backbone_channels[0], out_planes, 1, bias=False)
        # add MSRA initializations
        nn.init.kaiming_normal_(self.conv_out.weight, a=0, nonlinearity="relu")

    def forward(self, data, ic_resnet_blocks, label=None):
        p12, out_3 = self.CFF1(ic_resnet_blocks[2], ic_resnet_blocks[1])
        p012, out_2 = self.CFF2(p12, ic_resnet_blocks[0])
        out_1 = F.interpolate(p012, scale_factor=2)
        out_1 = self.conv_out(out_1)
        out_0 = F.interpolate(out_1, scale_factor=4)
        return out_0, out_1, out_2, out_3


class CascadeFeatureFusion(nn.Module):
    def __init__(self, in_planes, out_planes, norm_layer=nn.BatchNorm2d, n_classes=19):
        super(CascadeFeatureFusion, self).__init__()
        self.conv_1 = nn.Conv2d(in_planes[0], out_planes, 3, 1, 2, 2, bias=False)
        self.bn_1 = norm_layer(out_planes)
        self.conv_2 = nn.Conv2d(in_planes[1], out_planes, 3, 1, 1, 1, bias=False)
        self.bn_2 = norm_layer(out_planes)
        self.conv_3 = nn.Conv2d(in_planes[0], n_classes, 1, bias=False)
        # add MSRA initializations
        nn.init.kaiming_normal_(self.conv_1.weight, a=0, nonlinearity="relu")
        nn.init.kaiming_normal_(self.conv_2.weight, a=0, nonlinearity="relu")
        nn.init.kaiming_normal_(self.conv_3.weight, a=0, nonlinearity="relu")

    def forward(self, x1, x2):
        x1_up = F.interpolate(x1, scale_factor=2, mode="bilinear")
        x1 = self.conv_1(x1_up)
        x1 = self.bn_1(x1)
        x2 = self.conv_2(x2)
        x2 = self.bn_2(x2)

        loss_out = self.conv_3(x1_up) if self.training else None
        return F.relu(x1 + x2, inplace=True), loss_out


class ICNet_wrapper(BaseNet):
    def __init__(self, num_classes, backbone, aux=False, se_loss=False, **kwargs):
        super().__init__(num_classes, backbone, aux, se_loss, **kwargs)
        self.head = ICNet(num_classes, self.pretrained.base_inchannels)
        self.aux_indexes = [1, 2]
        self.num_outputs = 3

    def forward(self, img):
        ic_resnet_blocks = self.base_forward(img)
        results = self.head(img, list(ic_resnet_blocks))
        return results


def get_icnet(
    dataset="", backbone="", pretrained=False, root="~/.encoding/models", **kwargs
):
    return ICNet_wrapper(
        datasets[dataset.lower()].NUM_CLASS,
        backbone,
        pretrained=pretrained,
        root=root,
        **kwargs,
    )
