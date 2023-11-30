###########################################################################
# Created by: Hang Zhang
# Email: zhang.hang@rutgers.edu
# Copyright (c) 2017
###########################################################################

import os
import numpy as np
from tqdm import tqdm
from torch.nn import Parameter
import torch
from torch.utils import data
import torchvision.transforms as transform
from torch.nn.parallel.scatter_gather import gather
import time

import sys

sys.path.insert(0, "../../")

import encoding.utils as utils
from encoding.nn import SegmentationLosses, BatchNorm2d
from encoding.parallel import DataParallelModel, DataParallelCriterion
from encoding_custom.datasets import get_segmentation_dataset, test_batchify_fn
from encoding_custom.models import get_model, get_segmentation_model, MultiEvalModule

from option import Options

torch_ver = torch.__version__[:3]
if torch_ver == "0.3":
    from torch.autograd import Variable


def test(args):
    # output folder
    outdir = "outdir"
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    # data transforms
    input_transform = transform.Compose(
        [
            transform.ToTensor(),
            transform.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    # dataset
    if args.eval:
        testset = get_segmentation_dataset(
            args.dataset, split="val", mode="val", transform=input_transform
        )
    else:
        testset = get_segmentation_dataset(
            args.dataset, split="test", mode="test", transform=input_transform
        )
    # dataloader
    loader_kwargs = (
        {"num_workers": args.workers, "pin_memory": True} if args.cuda else {}
    )
    test_data = data.DataLoader(
        testset,
        batch_size=args.test_batch_size,
        drop_last=False,
        shuffle=False,
        collate_fn=test_batchify_fn,
        **loader_kwargs
    )
    # model
    if args.model_zoo is not None:
        model = get_model(args.model_zoo, pretrained=True)
    else:
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
        # resuming checkpoint
        # if args.resume is None or not os.path.isfile(args.resume):
        #    raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume))
        # checkpoint = torch.load(args.resume)
        # strict=False, so that it is compatible with old pytorch saved models
        # pretrained_dict = checkpoint['state_dict']
        # model_dict = model.state_dict()

        # for name, param in pretrained_dict.items():
        #    if name not in model_dict:
        #        continue
        #    if isinstance(param, Parameter):
        # backwards compatibility for serialized parameters
        #        param = param.data
        #    model_dict[name].copy_(param)

        # model.load_state_dict(checkpoint['state_dict'])
        # print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))

    # print(model)

    model = model.cuda()
    model.eval()

    run_time = list()

    for i in range(0, 100):
        input = torch.randn(1, 3, 1024, 384).cuda()
        # ensure that context initialization and normal_() operations
        # finish before you start measuring time
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        start = time.perf_counter()

        with torch.no_grad():
            output = model(input)

        torch.cuda.synchronize()  # wait for mm to finish
        end = time.perf_counter()

        print(end - start)

        run_time.append(end - start)

    run_time.pop(0)

    print("Mean fps is ", 1.0 / np.mean(run_time))


if __name__ == "__main__":
    args = Options().parse()
    # args.model='pspnet'
    args.aux = False
    args.se_loss = False
    torch.manual_seed(args.seed)
    args.test_batch_size = torch.cuda.device_count()
    test(args)
