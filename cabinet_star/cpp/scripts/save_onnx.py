###########################################################################
# Created by: Andrei - Doru Pata
# Email: andrei.pata@navinfo.eu
# Copyright (c) 2019
#
# Save CabiNet Pytorch model to ONNX format.
# Output: `../generated/cabinet.onnx`
# Note: In case of `upsample` error comment following line from `encoding_custom/models/fast_laddernet_se.py`:
#     `x[0] = F.upsample(x[0], imsize, **self._up_kwargs)`
#
# Example: python save_onnx.py --resume=/data/output/shabbir/cleaned_up/runs/mapillary/shelfnet/cabinet_base_mapillary_resnet101_0012/model_best.pth.tar --data-folder=/data/input/datasets/Mapillary_v1.1
###########################################################################
import json
import os
import torch
import torch.onnx
from torch.nn import BatchNorm2d

from encoding_custom.models import get_segmentation_model
from experiments.segmentation.option import Options


def save(args):
    model = get_segmentation_model(
        args.model,
        dataset=args.dataset,
        backbone=args.backbone,
        aux=args.aux,
        se_loss=args.se_loss,
        norm_layer=BatchNorm2d,
        base_size=args.base_size,
        crop_size=args.crop_size,
    )
    # Resuming checkpoint
    if args.resume is None or not os.path.isfile(args.resume):
        raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
    checkpoint = torch.load(args.resume)
    # strict=False, so that it is compatible with old pytorch saved models
    if "state_dict" in checkpoint.keys():
        model.load_state_dict(checkpoint["state_dict"])
    elif "model" in checkpoint.keys():
        model.load_state_dict(checkpoint["model"])
    else:
        raise ("loaded checkpoint has no params key!")
    print(
        "=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint["epoch"])
    )

    # print(model)

    dummy_input = torch.randn(1, 3, 224, 224).cuda()
    model.cuda()
    torch.onnx.export(model, dummy_input, "../generated/cabinet.onnx", verbose=True)


if __name__ == "__main__":
    args = Options().parse()
    args.custom_dict = json.loads(args.custom_dict)
    torch.manual_seed(args.seed)
    save(args)
