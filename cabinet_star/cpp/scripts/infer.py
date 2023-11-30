###########################################################################
# Created by: Andrei - Doru Pata
# Email: andrei.pata@navinfo.eu
# Copyright (c) 2019
#
# Run CabiNet model and save various tensors to be used to validate C++ model.
#
# python infer.py --resume=/home/backup/models/cabinet/shabbir/cleaned_up/runs/mapillary/shelfnet/cabinet_base_mapillary_resnet101_0012/checkpoint_150.pth.tar --data-folder=/home/backup/data/cabinet/mapillary/
###########################################################################
from cupy.cuda import nvtx
from matplotlib import pyplot as plt
import os
import sys
import torch
from torch.nn import BatchNorm2d
from torch.nn import functional as func
from torch.utils import data
import torchvision.transforms as transform
from tqdm import tqdm

sys.path.insert(0, "../../")
from encoding_custom.datasets import get_segmentation_dataset, test_batchify_fn
from encoding_custom.models import get_segmentation_model
from experiments.segmentation.option import Options

profiler_enabled = True


def profiler_push(enabled=True, *args, **kwargs):
    if enabled:
        nvtx.RangePush(*args, **kwargs)


def profiler_pop(enabled=True):
    if enabled:
        nvtx.RangePop()


def crop_by_factor(img, factor):
    height_crop = (img.shape[1] // factor) * factor
    width_crop = (img.shape[2] // factor) * factor

    diff_hight = img.shape[1] - height_crop
    diff_width = img.shape[2] - width_crop

    offset_height = diff_hight // 2
    offset_width = diff_width // 2

    return img[
        :,
        offset_height : height_crop + offset_height,
        offset_width : width_crop + offset_width,
    ]


def resize(img, target_size):
    profiler_push(profiler_enabled, "resize", 0)
    ratio = img.shape[2] / img.shape[1]
    cols_crop = target_size
    rows_crop = target_size
    offset_cols = 0
    offset_rows = 0

    pref_diff_cols = abs(img.shape[2] - target_size)
    pref_diff_rows = abs(img.shape[1] - target_size)

    img = torch.unsqueeze(img, 0)
    if pref_diff_cols < pref_diff_rows:
        adj_size_height = target_size // ratio
        img = func.interpolate(img, (adj_size_height, target_size), mode="bilinear")

        rows_crop = (adj_size_height // 32) * 32
        offset_rows = (adj_size_height - rows_crop) // 2
    else:
        adj_size_width = int(target_size * ratio)
        img = func.interpolate(img, (target_size, adj_size_width), mode="bilinear")

        cols_crop = (adj_size_width // 32) * 32
        offset_cols = (adj_size_width - cols_crop) // 2

    img = img[
        :,
        :,
        offset_rows : offset_rows + rows_crop,
        offset_cols : offset_cols + cols_crop,
    ]

    profiler_pop(profiler_enabled)  # "resize"
    return img


def test(args):
    # output folder
    outdir = args.save_dir
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
    testset = get_segmentation_dataset(
        args.dataset,
        split="test",
        mode="test",
        transform=input_transform,
        root=args.data_folder,
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
    if args.resume is None or not os.path.isfile(args.resume):
        raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
    checkpoint = torch.load(args.resume)
    # strict=False, so that it is compatible with old pytorch saved models
    if "state_dict" in checkpoint.keys():
        model.load_state_dict(checkpoint["state_dict"], strict=False)
    elif "model" in checkpoint.keys():
        model.load_state_dict(checkpoint["model"], strict=False)
    else:
        raise ("loaded checkpoint has no params key!")
    print(
        "=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint["epoch"])
    )

    # print(model)
    model.eval()
    model.cuda()

    tbar = tqdm(test_data)
    for i, (img, dst) in enumerate(tbar):
        profiler_push(profiler_enabled, "img" + str(i), 0)

        with torch.no_grad():
            # cur = image[0]
            # plt.imshow(image[0].permute(1, 2, 0))
            # plt.show()

            adj_img = resize(img[0], 1024)

            profiler_push(profiler_enabled, "forward", 2)
            outputs = model(adj_img.cuda())
            profiler_pop(profiler_enabled)  # "forward"

            profiler_push(profiler_enabled, "post-process", 2)
            predicts = [
                torch.max(output, 1)[1].cpu().numpy().squeeze() for output in outputs
            ]
            profiler_pop(profiler_enabled)  # "post-process"

        profiler_push(profiler_enabled, "color", 2)
        for predict, impath in zip(predicts, dst):
            mask = testset.apply_color_map(predict)
            outname = os.path.splitext(impath)[0] + ".png"
            mask.save(os.path.join(outdir, outname))
        profiler_pop(profiler_enabled)  # "color"

        profiler_pop(profiler_enabled)  # "image" + str(i)


if __name__ == "__main__":
    args = Options().parse()
    torch.manual_seed(args.seed)
    test(args)
