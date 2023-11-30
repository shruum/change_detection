import os

import random
from PIL import Image, ImageOps

from encoding.datasets.cityscapes import CitySegmentation
import matplotlib.pyplot as plt


class CityScapesRect(CitySegmentation):
    def __init__(
        self,
        root=os.path.expanduser("~/.encoding/data"),
        split="train",
        mode=None,
        transform=None,
        target_transform=None,
        scale=1,
        **kwargs
    ):
        super(CityScapesRect, self).__init__(
            root, split, mode, transform, target_transform, **kwargs
        )
        self.scale = scale

    def _val_sync_transform(self, img, mask):
        w, h = img.size
        ow = w // self.scale
        oh = h // self.scale
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # final transform
        return img, self._mask_transform(mask)

    def _sync_transform(self, img, mask):
        # random mirror
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        w, h = img.size
        ow = w // self.scale
        oh = h // self.scale
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # final transform
        return img, self._mask_transform(mask)
