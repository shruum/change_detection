###########################################################################
# Created by: Andrei - Doru Pata
# Email: andrei.pata@navinfo.eu
# Copyright (c) 2019
#
# Save PyTorch model to TorchScript format (".pt").
# Output: `../runs/python/models/<args.model>.pt`
#
# Example: python save_torchscript.py
# --model=tascnet
# --resume=/home/backup/models/cabinet/shabbir/cleaned_up/runs/mapillary/tascnet/tascnet_1536_fpn_resnet50_0012/model_best.pth.tar
# --data-folder=/home/backup/data/cabinet/mapillary/
# --backbone=fpn_resnet50
# --crop-size=1024
# --base-size=1024
###########################################################################
import os
import sys
import torch
from torch.nn import BatchNorm2d

sys.path.insert(0, "../../")
from encoding_custom.models import get_segmentation_model
from experiments.segmentation.option import Options


def save(args):
    # Model
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

    # Resume checkpoint
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

    model.eval()

    with torch.no_grad():
        if args.no_cuda:
            example = torch.rand(1, 3, 1024, 1024)
        else:
            example = torch.rand(1, 3, 1024, 1024).cuda()
            model.cuda()

    traced_script_module = torch.jit.trace(model, example)
    traced_script_module.save("../runs/python/models/" + args.model + ".pt")


if __name__ == "__main__":
    args = Options().parse()
    torch.manual_seed(args.seed)
    save(args)
