# ------------------------------------------------------------------------------
# Portions of this code are from
# CornerNet (https://github.com/princeton-vl/CornerNet)
# Copyright (c) 2019, NavInfo Europe
# NOTE:Implementation of CenterNet Head in Detection as a service Framework
# ------------------------------------------------------------------------------
import torch
import math
import torch.nn as nn
from od.modeling.head.centernet.DCNv2.dcn_v2 import DCN


class CenterNetHeadPredictor(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # ------------ parsing the CenterNet Head from configuration file ------------------#
        self.inplanes = cfg.MODEL.BACKBONE.OUT_CHANNEL
        self.backbone_feature = cfg.MODEL.HEAD.BACKBONE_FEATURE
        self.num_deconv_layers = cfg.MODEL.HEAD.NUM_DECONV_LAYERS
        self.deconv_layer_config = cfg.MODEL.HEAD.DECONV_LAYER_CONFIG
        self.deconv_kernel_config = cfg.MODEL.HEAD.DECONV_KERNEL
        self.heads = cfg.MODEL.HEAD.HEAD_CONFIG
        self.head_conv = cfg.MODEL.HEAD.HEAD_CONV
        self.deconv_with_bias = False

        # creating the deconv_blocks
        self.deconv_layers = self._make_deconv_layer(
            self.num_deconv_layers, self.deconv_layer_config, self.deconv_kernel_config
        )

        # now creating the head
        for head in sorted(self.heads):
            num_output = self.heads[head]
            if self.head_conv > 0:
                fc = nn.Sequential(
                    nn.Conv2d(
                        in_channels=self.deconv_layer_config[
                            len(self.deconv_layer_config) - 1
                        ],
                        out_channels=self.head_conv,
                        kernel_size=3,
                        padding=1,
                        bias=True,
                    ),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(
                        in_channels=self.head_conv,
                        out_channels=num_output,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                    ),
                )
                if "hm" in head:
                    fc[-1].bias.data.fill_(-2.19)
                else:
                    self.fill_fc_weights(fc)
            else:
                fc = nn.Conv2d(
                    in_channels=64,
                    out_channels=num_output,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=True,
                )
                if "hm" in head:
                    fc.bias.data.fill_(-2.19)
                else:
                    self.fill_fc_weights(fc)
            self.__setattr__(head, fc)
        self.init_weights(pretrained=True)

    # forward function to connect the deconv layers and the 3 heads
    def forward(self, x):
        if self.backbone_feature > x.size(2):
            diff = (self.backbone_feature - x.size(2)) // 2
            pad_inp = nn.ConstantPad2d((diff, diff, diff, diff), 0)
            x = pad_inp(x)
        x = self.deconv_layers(x)
        ret = {}
        for head in self.heads:
            ret[head] = self.__getattr__(head)(x)
        return [ret]

    def _make_deconv_layer(
        self, num_deconv_layers, deconv_layer_config, deconv_kernel_config
    ):
        # internal function to configure the deconv layer
        def _get_deconv_cfg(deconv_kernel):
            if deconv_kernel == 4:
                padding = 1
                output_padding = 0
            elif deconv_kernel == 3:
                padding = 1
                output_padding = 1
            elif deconv_kernel == 2:
                padding = 0
                output_padding = 0
            return deconv_kernel, padding, output_padding

        layers = []
        for i in range(num_deconv_layers):
            # checking assertion that the layer and kernel configs match
            assert num_deconv_layers == len(
                deconv_layer_config
            ), "ERROR: num_deconv_layers is different from len(deconv_layer_config). Check config file."
            assert num_deconv_layers == len(
                deconv_kernel_config
            ), "ERROR: num_deconv_layers is different len(deconv_kernel_config). Check config file."

            # obtaining the kernel, padding and output padding configuration to build the deconv layer
            kernel, padding, output_padding = _get_deconv_cfg(
                deconv_kernel=deconv_kernel_config[i]
            )

            # getting the number of output channels from deconv layer configuration
            planes = deconv_layer_config[i]

            # adding the deformable convolution
            fc = DCN(
                self.inplanes,
                planes,
                kernel_size=(3, 3),
                stride=1,
                padding=1,
                dilation=1,
                deformable_groups=1,
            )

            # adding the conv2d transpose
            upconv = nn.ConvTranspose2d(
                in_channels=planes,
                out_channels=planes,
                kernel_size=kernel,
                stride=2,
                padding=padding,
                output_padding=output_padding,
                bias=self.deconv_with_bias,
            )

            # filling up the weights
            self.fill_up_weights(upconv)

            layers.append(fc)
            layers.append(nn.BatchNorm2d(planes, momentum=0.1))
            layers.append(nn.ReLU(inplace=True))
            layers.append(upconv)
            layers.append(nn.BatchNorm2d(planes, momentum=0.1))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes
        return nn.Sequential(*layers)

    def init_weights(self, pretrained=True):
        if pretrained:
            for name, m in self.deconv_layers.named_modules():
                if isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def fill_up_weights(self, upconv):
        w = upconv.weight.data
        f = math.ceil(w.size(2) / 2)
        c = (2 * f - 1 - f % 2) / (2.0 * f)
        for i in range(w.size(2)):
            for j in range(w.size(3)):
                w[0, 0, i, j] = (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
        for c in range(1, w.size(0)):
            w[c, 0, :, :] = w[0, 0, :, :]

    def fill_fc_weights(self, layers):
        for m in layers.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                # torch.nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
                # torch.nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
