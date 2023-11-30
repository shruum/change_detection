import torch
import torch.nn as nn
import torch.nn.functional as F
from encoding_custom.datasets import datasets
from encoding_custom.models.base import BaseNet
from encoding_custom.models.LadderNetv66_small import BasicBlock


class BottleneckAUX(nn.Module):
    """ResNet Bottleneck"""

    # pylint: disable=unused-argument
    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        dilation=1,
        previous_dilation=1,
        norm_layer=None,
    ):
        super(BottleneckAUX, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(planes)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=False,
        )
        self.bn2 = norm_layer(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.dilation = dilation
        self.stride = stride

    def _sum_each(self, x, y):
        assert len(x) == len(y)
        z = []
        for i in range(len(x)):
            z.append(x[i] + y[i])
        return z

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)

        return out


def get_channels_list(backbone):
    if "fpn" in backbone:
        return [256, 256, 256, 256]
    elif "res" in backbone:
        return [256, 256 * 2, 256 * 2 ** 2, 256 * 2 ** 3]
    elif "inception" in backbone:
        return [192, 384, 1024, 1536]


class DPM(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.up_conv_1 = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
            bias=True,
        )

        self.up_conv_2 = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
            bias=True,
        )

        self.down_1 = nn.Conv2d(
            in_channels, in_channels, stride=2, kernel_size=3, padding=1
        )
        self.down_2 = nn.Conv2d(
            in_channels, in_channels, stride=2, kernel_size=3, padding=1
        )

    def forward(self, x):
        x_up_1 = self.up_conv_1(x)
        x_up_2 = self.up_conv_1(x_up_1)
        x_down_1 = self.down_1(x_up_2)
        x_down_2 = x_down_1 + x_up_1
        x_down_2 = self.down_2(x_down_2)
        x_out = x_down_2 + x
        return x_out


class Decoder(nn.Module):
    def __init__(
        self,
        planes,
        layers,
        kernel=3,
        block=BasicBlock,
        use_dense=False,
        norm_layer=None,
    ):
        super().__init__()

        self.planes = planes
        self.layers = layers
        self.kernel = kernel
        self.padding = int((kernel - 1) / 2)
        # self.inconv = block(planes[0], planes[0], norm_layer=norm_layer) # need to remove (might break resume method)
        # create module for bottom block
        self.bottom = block(planes[-1], planes[-1], norm_layer=norm_layer,)

        # create module list for up branch
        self.up_conv_list = nn.ModuleList()
        self.up_dense_list = nn.ModuleList()
        for i in range(0, layers - 1):
            self.up_conv_list.append(
                nn.ConvTranspose2d(
                    planes[(layers - 1 - i)],
                    planes[max(0, layers - i - 2)],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                    bias=True,
                )
            )
            self.up_dense_list.append(
                block(
                    planes[max(0, layers - i - 2)],
                    planes[max(0, layers - i - 2)],
                    norm_layer=norm_layer,
                )
            )
            # 256, 128, 64

        self.down_conv_list = nn.ModuleList()
        for i in range(0, layers - 1):
            self.down_conv_list.append(
                nn.Conv2d(
                    planes[i],
                    planes[(i + 1)],
                    stride=2,
                    kernel_size=kernel,
                    padding=self.padding,
                )
            )

    def forward(self, x):
        # bottom branch
        downsampled = []
        for i in range(0, self.layers - 1):
            out = self.down_conv_list[i](x[i])
            out = F.relu(out)
            downsampled.append(out)

        out = self.bottom(x[-1] + downsampled[-1])
        bottom = out

        # up branch
        up_out = []
        up_out.append(bottom)

        for j in range(0, self.layers - 1):
            out = self.up_conv_list[j](out) + x[self.layers - j - 2]
            if (self.layers - j - 3) >= 0:
                out += downsampled[self.layers - j - 3]
            # out = F.relu(out)
            out = self.up_dense_list[j](out)
            up_out.append(out)

        return up_out


class Cabinet(BaseNet):
    def __init__(
        self,
        num_classes,
        backbone=None,
        aux=False,
        se_loss=True,
        lateral=False,
        norm_layer=None,
        dilated=False,
        multiscale=False,
        use_dense=False,
        add_another_encoder=False,
        base_outchannels=None,
        **kwargs
    ):
        super(Cabinet, self).__init__(
            num_classes,
            backbone,
            aux,
            se_loss,
            norm_layer=norm_layer,
            dilated=dilated,
            **kwargs
        )
        base_inchannels = (
            self.pretrained.base_inchannels
            if hasattr(self.pretrained, "base_inchannels")
            else get_channels_list(backbone)
        )
        if base_outchannels is None:
            base_outchannels = base_inchannels

        self.tensorrt_flag = False
        self.tensorrt_5_flag = False

        self.head = LadderHead(
            base_inchannels=base_inchannels,
            base_outchannels=base_outchannels,
            out_channels=num_classes,
            norm_layer=norm_layer,
            use_dense=use_dense,
            add_another_encoder=add_another_encoder,
            upkwargs=self._up_kwargs,
        )
        if aux:
            self.aux_indexes = [1, 2, 3]
            self.num_outputs = 4
            self.auxlayer = nn.ModuleList()
            self.auxlayer.append(
                nn.Sequential(
                    BottleneckAUX(
                        base_outchannels[1],
                        base_outchannels[0] // 4,
                        norm_layer=norm_layer,
                    ),
                    nn.Conv2d(base_outchannels[0], num_classes, 1)
                    # nn.Dropout2d(0.1, False), nn.Conv2d(base_outchannels[1], num_classes, 1))) prev aux_layer
                )
            )

            self.auxlayer.append(
                nn.Sequential(
                    BottleneckAUX(
                        base_outchannels[2],
                        base_outchannels[0] // 4,
                        norm_layer=norm_layer,
                    ),
                    nn.Conv2d(base_outchannels[0], num_classes, 1)
                    # nn.Dropout2d(0.1, False), nn.Conv2d(base_outchannels[2], num_classes, 1))) prev aux_layer
                )
            )
            self.auxlayer.append(
                nn.Sequential(
                    BottleneckAUX(
                        base_outchannels[3],
                        base_outchannels[0] // 4,
                        norm_layer=norm_layer,
                    ),
                    nn.Conv2d(base_outchannels[0], num_classes, 1)
                    # nn.Dropout2d(0.1, False), nn.Conv2d(base_outchannels[3], num_classes, 1))) prev aux_layer
                )
            )

        # base_forward

    def forward(self, x):
        imsize = x.size()[2:]

        class BilinearUpsample(torch.autograd.Function):
            @staticmethod
            def symbolic(g, input):
                return g.op(
                    "resize_bilinear_TRT",
                    input,
                    output_size_i=imsize,
                    align_corners_i=True,
                )

            @staticmethod
            def forward(ctx, x):
                return F.upsample(x, size=imsize, mode="bilinear", align_corners=True)

        features = self.base_forward(x)  # return 4 features from restnet backbone
        x_out, rest_of_features = self.head(features)
        x_out = list(x_out)

        if self.tensorrt_flag:
            x_out[0] = F.upsample(
                x_out[0], (self.tensorrt_heiget, self.tensorrt_width), **self._up_kwargs
            )
        elif self.tensorrt_5_flag:
            x_out[0] = BilinearUpsample().apply(x_out[0])
        else:
            x_out[0] = F.upsample(x_out[0], imsize, **self._up_kwargs)

        if hasattr(self, "auxlayer"):
            x_out.append(F.upsample(self.auxlayer[0](rest_of_features[-2]), imsize))
            x_out.append(F.upsample(self.auxlayer[1](rest_of_features[-3]), imsize))
            x_out.append(F.upsample(self.auxlayer[2](rest_of_features[-4]), imsize))

        return tuple(x_out)


class Encoder(nn.Module):
    def __init__(self, base_outchannels, norm_layer=None):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=base_outchannels,
            out_channels=base_outchannels,
            kernel_size=1,
            bias=False,
        )
        self.conv2 = nn.Conv2d(
            in_channels=base_outchannels * 2,
            out_channels=base_outchannels * 2,
            kernel_size=1,
            bias=False,
        )
        self.conv3 = nn.Conv2d(
            in_channels=base_outchannels * 2 ** 2,
            out_channels=base_outchannels * 2 ** 2,
            kernel_size=1,
            bias=False,
        )
        self.conv4 = nn.Conv2d(
            in_channels=base_outchannels * 2 ** 3,
            out_channels=base_outchannels * 2 ** 3,
            kernel_size=1,
            bias=False,
        )

        self.bn1 = norm_layer(base_outchannels)
        self.bn2 = norm_layer(base_outchannels * 2)
        self.bn3 = norm_layer(base_outchannels * 2 ** 2)
        self.bn4 = norm_layer(base_outchannels * 2 ** 3)

        self.down_1 = nn.Conv2d(
            base_outchannels, base_outchannels * 2, stride=2, kernel_size=3, padding=1,
        )

        self.down_2 = nn.Conv2d(
            base_outchannels * 2,
            base_outchannels * 2 ** 2,
            stride=2,
            kernel_size=3,
            padding=1,
        )

        self.down_3 = nn.Conv2d(
            base_outchannels * 2 ** 2,
            base_outchannels * 2 ** 3,
            stride=2,
            kernel_size=3,
            padding=1,
        )

    def forward(self, x):
        x1, x2, x3, x4 = x

        out1 = self.conv1(x1)  # 64 --> 64
        out1 = self.bn1(out1)
        out1 = F.relu(out1)

        out_down_1 = self.down_1(x1)
        out_down_1 = F.relu(out_down_1)

        out2 = self.conv2(x2)  # 128 --> 128
        out2 = self.bn2(out2)
        out2 = F.relu(out2) + out_down_1

        out_down_2 = self.down_2(x2)
        out_down_2 = F.relu(out_down_2)

        out3 = self.conv3(x3)  # 256 --> 256
        out3 = self.bn3(out3)
        out3 = F.relu(out3) + out_down_2

        out_down_3 = self.down_3(x3)
        out_down_3 = F.relu(out_down_3)

        out4 = self.conv4(x4)  # 512 --> 512
        out4 = self.bn4(out4)
        out4 = F.relu(out4) + out_down_3

        return [out1, out2, out3, out4]


class LadderHead(nn.Module):
    def __init__(
        self,
        base_inchannels,
        base_outchannels,
        out_channels,
        norm_layer,
        use_dense,
        add_another_encoder,
        upkwargs,
    ):
        super(LadderHead, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=base_inchannels[0],
            out_channels=base_outchannels[0],
            kernel_size=1,
            bias=False,
        )
        self.conv2 = nn.Conv2d(
            in_channels=base_inchannels[1],
            out_channels=base_outchannels[1],
            kernel_size=1,
            bias=False,
        )
        self.conv3 = nn.Conv2d(
            in_channels=base_inchannels[2],
            out_channels=base_outchannels[2],
            kernel_size=1,
            bias=False,
        )
        self.conv4 = nn.Conv2d(
            in_channels=base_inchannels[3],
            out_channels=base_outchannels[3],
            kernel_size=1,
            bias=False,
        )

        self.bn1 = norm_layer(base_outchannels[0])
        self.bn2 = norm_layer(base_outchannels[1])
        self.bn3 = norm_layer(base_outchannels[2])
        self.bn4 = norm_layer(base_outchannels[3])
        # if add_another_encoder:
        #     self.encoder = Encoder(base_outchannels=base_outchannels, norm_layer=norm_layer)
        self.decoder = Decoder(
            planes=base_outchannels,
            layers=4,
            norm_layer=norm_layer,
            use_dense=use_dense,
        )
        self.final = nn.Conv2d(base_outchannels[0], out_channels, 1)

    def forward(self, x):
        x1, x2, x3, x4 = x

        out1 = self.conv1(x1)  # 256 --> 64
        out1 = self.bn1(out1)
        out1 = F.relu(out1)

        out2 = self.conv2(x2)  # 512 --> 128
        out2 = self.bn2(out2)
        out2 = F.relu(out2)

        out3 = self.conv3(x3)  # 1024 --> 256
        out3 = self.bn3(out3)
        out3 = F.relu(out3)

        out4 = self.conv4(x4)  # 2048 --> 512
        out4 = self.bn4(out4)
        out4 = F.relu(out4)

        # if hasattr(self, 'encoder'):
        #     out1, out2, out3, out4 = self.encoder([out1, out2, out3, out4])
        out = self.decoder([out1, out2, out3, out4])
        pred = [self.final(out[-1])]

        # if self.se_loss:
        #     enc = F.max_pool2d(out[0], kernel_size=out[0].size()[2:])
        #     enc = torch.squeeze(enc, -1)
        #     enc = torch.squeeze(enc, -1)
        #     se = self.selayer(enc)
        #     pred.append(se)

        return pred, out


def get_rgpnet(
    dataset="pascal_voc",
    backbone="resnet50",
    pretrained=False,
    root="~/.encoding/models",
    **kwargs
):
    if type(dataset) == str:
        num_class = datasets[dataset.lower()].NUM_CLASS
    elif type(dataset) == int:
        num_class = dataset
    else:
        raise Exception("dataset is ill defined")
    kwargs["pretrained"] = pretrained
    model = Cabinet(
        num_class,
        backbone,
        use_dense=False,
        base_outchannels=[64, 64 * 2, 64 * 2 ** 2, 64 * 2 ** 3],
        **kwargs
    )
    return model


def get_cabinet_v4_big(
    dataset="pascal_voc",
    backbone="resnet50",
    pretrained=False,
    root="~/.encoding/models",
    **kwargs
):

    model = Cabinet(
        datasets[dataset.lower()].NUM_CLASS,
        backbone,
        use_dense=False,
        base_outchannels=[256, 256 * 2, 256 * 2 ** 2, 256 * 2 ** 3],
        **kwargs
    )
    return model


def get_cabinet_v5(
    dataset="pascal_voc",
    backbone="resnet50",
    pretrained=False,
    root="~/.encoding/models",
    **kwargs
):

    model = Cabinet(
        datasets[dataset.lower()].NUM_CLASS, backbone, use_dense=True, **kwargs
    )
    return model


def get_cabinet_v6(
    dataset="pascal_voc",
    backbone="resnet50",
    pretrained=False,
    root="~/.encoding/models",
    **kwargs
):

    model = Cabinet(
        datasets[dataset.lower()].NUM_CLASS,
        backbone,
        use_dense=False,
        add_another_encoder=True,
        **kwargs
    )
    return model


def get_cabinet_slim(
    dataset="pascal_voc",
    backbone="resnet50",
    pretrained=False,
    root="~/.encoding/models",
    **kwargs
):

    model = Cabinet(
        datasets[dataset.lower()].NUM_CLASS,
        backbone,
        use_dense=False,
        base_outchannels=None,
        **kwargs
    )
    return model


if __name__ == "__main__":
    A = torch.rand(2, 3, 1024, 1024).cuda()
    net = get_rgpnet("mapillary", "resnet101", norm_layer=torch.nn.BatchNorm2d).cuda()
    out = net(A)
    print(out[0].shape)
