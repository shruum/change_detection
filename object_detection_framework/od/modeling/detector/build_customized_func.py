import torch
from torch import nn


class build_customized_func(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def forward(self, images):

        if self.cfg.MODEL.HEAD.NAME == "ThunderNetHead":

            def get_im_info(images):
                im_info = torch.Tensor((images.size(2), images.size(3), 0)).to(
                    images.get_device(), dtype=torch.float32
                )

                if self.cfg.EXPORT == "onnx":
                    im_info = torch.stack([im_info] * int(images.size(0)))
                else:
                    im_info = torch.stack([im_info] * images.size(0))

                return dict(im_info=im_info)

            return get_im_info(images)

        return dict()
