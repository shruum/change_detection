###########################################################################
# Created by: Hang Zhang
# Email: zhang.hang@rutgers.edu
# Copyright (c) 2017
###########################################################################


import os
import copy
import numpy as np
from tqdm import tqdm
from torch.nn import Parameter
import torch
from torch.utils import data
import torchvision.transforms as transform
from torch.nn.parallel.scatter_gather import gather
from torch import nn

import sys
from contextlib import ExitStack

sys.path.insert(0, "../../")

from encoding_custom.utils.lr_scheduler import LR_Scheduler

import encoding.utils as utils
from encoding_custom.nn.loss import (
    SegmentationLosses,
    ClassBalancedSegmentationLosses,
    ClassBalancedSegmentationLossesWithLabelRelaxation,
    OHEMSegmentationLosses,
)
from encoding_custom.nn.semseg_loss import SemsegCrossEntropy
from encoding_custom.datasets import get_segmentation_dataset
from encoding_custom.models import get_segmentation_model
from encoding_custom.nn.soft_dice_loss import SoftDiceLoss
from encoding_custom.nn.focal_loss import FocalLoss
from encoding_custom.utils.radam import RAdam
from encoding_custom.utils.energy_meter import EnergyMeter
from encoding_custom.nn.mixup import mixup_data, mixup_criterion
from experiments.segmentation.option import Options
from tensorboardX import SummaryWriter
from encoding.nn.syncbn import SyncBatchNorm
import glob

import socket
from datetime import datetime
import json
import shutil

from lr_finder import LRFinder


def save_checkpoint(state, directory, is_best, filename="checkpoint.pth.tar"):
    """Saves checkpoint to disk"""
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, directory + "model_best.pth.tar")


class FullModel(nn.Module):
    """Distribute the loss on multi-gpu to reduce
    the memory cost in the main gpu.
    You can check the following discussion.
    https://discuss.pytorch.org/t/dataparallel-imbalanced-memory-usage/22551/21
    """

    def __init__(self, model, losses, mixup=False):
        super(FullModel, self).__init__()
        self.model = model
        self.losses = losses
        self.mixup = mixup

    def forward(self, inputs, labels, mixup_lambda=0):
        if self.training and self.mixup:
            labels_a = labels[0]
            labels_b = labels[1]

        outputs = self.model(inputs)[0]

        loss = 0
        for curr_loss in self.losses.keys():
            if self.training and self.mixup and (curr_loss == "seg"):
                loss += mixup_criterion(
                    self.losses[curr_loss], outputs, labels_a, labels_b, mixup_lambda
                )
            else:
                loss += self.losses[curr_loss](outputs, labels)

        return outputs, torch.unsqueeze(loss, 0)


def evaluate_class_weights(trainloader, num_classes, c=1.02):
    if c is not None:
        class_weights = torch.zeros(num_classes)
        for _, labels in trainloader:
            for label in torch.unique(labels):
                if label != -1:
                    class_weights[label] += torch.sum(labels == label)
        class_weights /= torch.sum(class_weights)
        class_weights = 1 / (torch.log(c + class_weights))
    else:
        class_weights = None
    return class_weights


class Trainer:
    def __init__(self, args, directory):
        self.directory = directory
        current_time = datetime.now().strftime("%b%d_%H-%M-%S")
        self.log_dir_name = os.path.join(
            self.directory, current_time + "_" + socket.gethostname()
        )

        self.writer = SummaryWriter(log_dir=self.log_dir_name)
        print(f"logdir = {self.log_dir_name}")
        self.writer.add_scalar("train_step/loss", 0, 0)
        self.writer.add_scalar("train_step/mIoU", 0, 0)
        self.writer.add_scalar("train_step/pixAcc", 0, 0)
        self.writer.add_scalar("train_epoch/loss", 0, 0)
        self.writer.add_scalar("train_epoch/mIoU", 0, 0)
        self.writer.add_scalar("train_epoch/pixAcc", 0, 0)
        self.writer.add_scalar("val_epoch/mIoU", 0, 0)
        self.writer.add_scalar("val_epoch/pixAcc", 0, 0)
        self.writer.add_scalar("val_epoch/loss", 0, 0)

        self.args = args
        # data transforms
        if args.use_mapillary_norms:
            im_mean = [0.41738699, 0.45732192, 0.46886091]
            im_std = [0.25685097, 0.26509955, 0.29067996]
        else:
            im_mean = [0.485, 0.456, 0.406]
            im_std = [0.229, 0.224, 0.225]

        input_transform = transform.Compose(
            [transform.ToTensor(), transform.Normalize(im_mean, im_std)]
        )
        im_mean = torch.Tensor(im_mean)
        im_std = torch.Tensor(im_std)
        self.inverse_transform = transform.Normalize(-im_mean / im_std, 1.0 / im_std)

        # dataset
        data_kwargs = {
            "transform": input_transform,
            "base_size": args.base_size,
            "crop_size": args.crop_size,
            "root": args.data_folder,
        }
        if args.dataset == "citys_v2":
            trainset = get_segmentation_dataset(
                args.dataset, split="train", mode="train", scale=2, **data_kwargs
            )
            testset = get_segmentation_dataset(
                args.dataset, split="val", mode="val", scale=2, **data_kwargs
            )
        else:
            trainset = get_segmentation_dataset(
                args.dataset, split="train", mode="train", **data_kwargs
            )
            testset = get_segmentation_dataset(
                args.dataset, split="val", mode="val", **data_kwargs
            )

        # dataloader
        kwargs = {"num_workers": args.workers, "pin_memory": False} if args.cuda else {}
        self.trainloader = data.DataLoader(
            trainset, batch_size=args.batch_size, drop_last=True, shuffle=True, **kwargs
        )
        self.valloader = data.DataLoader(
            testset,
            batch_size=args.batch_size,
            drop_last=False,
            shuffle=False,
            **kwargs,
        )
        self.nclass = trainset.num_class

        # model
        if args.model == "swiftnet":
            assert (
                "single_scale" in args.backbone
            ), "backbone incompatible with swiftnet"
        model = get_segmentation_model(
            args.model,
            dataset=args.dataset,
            backbone=args.backbone,
            aux=args.aux,
            se_loss=args.se_loss,
            norm_layer=SyncBatchNorm,
            base_size=args.base_size,
            crop_size=args.crop_size,
            pretrained=args.pretrained,
            **args.custom_dict,
        )

        # count parameter number
        pytorch_total_params = sum(p.numel() for p in model.parameters())
        print("Total number of parameters: %d" % pytorch_total_params)

        # optimizer using different LR
        params_list = [{"params": model.pretrained.parameters(), "lr": args.lr}]
        if hasattr(model, "head"):
            if args.diflr:
                params_list.append(
                    {"params": model.head.parameters(), "lr": args.lr * 10}
                )
            else:
                params_list.append({"params": model.head.parameters(), "lr": args.lr})
        if hasattr(model, "auxlayer"):
            if args.diflr:
                params_list.append(
                    {"params": model.auxlayer.parameters(), "lr": args.lr * 10}
                )
            else:
                params_list.append(
                    {"params": model.auxlayer.parameters(), "lr": args.lr}
                )

        optimizer = None
        if args.optimizer.lower() == "sgd":
            optimizer = torch.optim.SGD(
                params_list,
                lr=args.lr,
                momentum=args.momentum,
                weight_decay=args.weight_decay,
            )

        elif args.optimizer.lower() == "adam":
            optimizer = torch.optim.Adam(
                params_list, lr=args.lr, weight_decay=args.weight_decay
            )

        elif args.optimizer.lower() == "rmsprop":
            optimizer = torch.optim.RMSprop(
                params_list,
                lr=args.lr,
                weight_decay=args.weight_decay,
                momentum=args.momentum,
            )

        elif args.optimizer.lower() == "radam":
            optimizer = RAdam(params_list, lr=args.lr, weight_decay=args.weight_decay)
        if args.lookAhead_steps is None:
            pass
        elif args.lookAhead_steps > 0:
            self.lookAhead_steps = args.lookAhead_steps
            self.lookAhead_alpha = args.lookAhead_alpha

        self.model, self.optimizer = model, optimizer

        # criterions
        self.losses = nn.ModuleDict()

        if hasattr(self.model, "num_outputs"):
            num_outputs = self.model.num_outputs
        else:
            num_outputs = 1

        if hasattr(self.model, "aux_indexes"):
            aux_indexes = self.model.aux_indexes
        else:
            aux_indexes = []

        if not self.args.no_cross:
            assert (
                args.model != "swiftnet"
            ), "default cross entropy not applicable to swiftnet"
            class_weights = evaluate_class_weights(
                self.trainloader, self.nclass, args.log_reweight
            )
            self.losses["seg"] = SegmentationLosses(
                se_loss=args.se_loss,
                aux=args.aux,
                nclass=self.nclass,
                num_outputs=num_outputs,
                aux_indexes=aux_indexes,
                weight=class_weights,
            )

        if args.model == "swiftnet":
            self.losses["seg"] = SemsegCrossEntropy(num_classes=self.nclass)

        if self.args.label_relaxed_loss:
            self.losses["seg"] = ClassBalancedSegmentationLossesWithLabelRelaxation(
                se_loss=args.se_loss,
                aux=args.aux,
                nclass=self.nclass,
                num_outputs=num_outputs,
                aux_indexes=aux_indexes,
                beta=args.class_balanced_beta,
            )

        if self.args.class_balanced_loss:
            self.losses["seg"] = ClassBalancedSegmentationLosses(
                se_loss=args.se_loss,
                aux=args.aux,
                nclass=self.nclass,
                num_outputs=num_outputs,
                aux_indexes=aux_indexes,
                beta=args.class_balanced_beta,
            )

        if self.args.soft_dice_loss:
            self.losses["soft_dice"] = SoftDiceLoss(self.nclass)

        if self.args.focal_loss:
            self.losses["focal"] = FocalLoss(alpha=None)
            # NOTE: for future: you can add class weight here,
            # e.g. for cityscape this would work already:
            # self.losses["focal"] = FocalLoss(alpha=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19])

        if self.args.ohem_loss:
            self.losses["ohem"] = OHEMSegmentationLosses(
                se_loss=args.se_loss, aux=args.aux, nclass=self.nclass
            )

        assert len(self.losses) > 0, "There should be at least one loss added"

        self.model = FullModel(
            self.model, self.losses, self.args.mixup_alpha is not None
        )

        # using cuda
        if args.cuda and args.multiple_GPUs:
            self.model = nn.DataParallel(self.model).cuda()

        elif args.cuda and not args.multiple_GPUs:
            self.model = self.model.cuda()

        self.best_pred = 0.0

        if args.resume_after_suspend:
            if os.path.exists(self.directory):
                files = glob.glob(os.path.join(self.directory, "checkpoint*.pth.tar"))
                if len(files) > 0:
                    files = sorted(files)
                    args.resume = files[-1]
                    args.ft = False

        # resuming checkpoint
        if args.resume is not None and len(args.resume) > 0:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint["epoch"]
            if args.cuda:
                # load weights for the same model
                # model and checkpoint have different strucutures
                pretrained_dict = checkpoint["state_dict"]
                model_dict = (
                    self.model.module.model.state_dict()
                    if args.multiple_GPUs
                    else self.model.model.state_dict()
                )

                for name, param in pretrained_dict.items():
                    if name not in model_dict:
                        print("No weights founds for %s layer in loaded model." % name)
                        continue
                    if isinstance(param, Parameter):
                        # backwards compatibility for serialized parameters
                        param = param.data
                    model_dict[name].copy_(param)

            else:
                self.model.module.model.load_state_dict(
                    checkpoint["state_dict"]
                ) if args.multiple_GPUs else self.model.model.load_state_dict(
                    checkpoint["state_dict"]
                )
            if not args.ft:
                self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.best_pred = checkpoint["best_pred"]
            print(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    args.resume, checkpoint["epoch"]
                )
            )

        # clear start epoch if fine-tuning
        if args.ft:
            args.start_epoch = 0
            self.best_pred = 0.0

        # lr scheduler

        if args.lr_scheduler == "clr":
            self.scheduler = LR_Scheduler(
                args.lr_scheduler,
                args.lr,
                args.epochs,
                len(self.trainloader),
                stepsize=len(self.trainloader) * args.clr_stepsize,
                max_lr=args.clr_max,
                rand_grad=args.rand_grad,
            )
        else:
            self.scheduler = LR_Scheduler(
                args.lr_scheduler,
                args.lr,
                args.epochs,
                len(self.trainloader),
                lr_step=args.lr_step,
                rand_grad=args.rand_grad,
            )

    def findlr(self):
        criterion = self.losses
        lr_finder = LRFinder(self.model.model, self.optimizer, criterion, device="cuda")
        lr_finder.range_test(
            self.trainloader, end_lr=10, num_iter=1000, step_mode="exp"
        )
        lr_finder.plot(log_lr=True)

    def training(self, epoch):
        # progressive resizing
        if self.args.epochs_per_resize is not None:
            if epoch < self.args.epochs_per_resize:
                self.trainloader.dataset.base_size = self.args.base_size // 8
                self.trainloader.dataset.crop_size = self.args.crop_size // 8
            elif epoch < self.args.epochs_per_resize * 2:
                self.trainloader.dataset.base_size = self.args.base_size // 4
                self.trainloader.dataset.crop_size = self.args.crop_size // 4
            elif epoch < self.args.epochs_per_resize * 3:
                self.trainloader.dataset.base_size = self.args.base_size // 2
                self.trainloader.dataset.crop_size = self.args.crop_size // 2
            else:
                self.trainloader.dataset.base_size = self.args.base_size
                self.trainloader.dataset.crop_size = self.args.crop_size

        # mini-batch loop
        train_loss = 0.0
        total_inter, total_union, total_correct, total_label = 0, 0, 0, 0
        tbar = tqdm(self.trainloader, ascii=True)
        if self.args.lookAhead_steps is None:
            pass
        elif self.args.lookAhead_steps != 0:
            prev_state = copy.deepcopy(self.model.module.state_dict())
        for i, (images, targets) in enumerate(tbar):
            self.model.train()
            lr = self.scheduler(self.optimizer, i, epoch, self.best_pred)
            self.optimizer.zero_grad()

            mixup_lambda = 0
            if self.args.mixup_alpha is not None:
                images, targets_a, targets_b, mixup_lambda = mixup_data(
                    images, targets, self.args.mixup_alpha
                )
                train_targets = torch.stack((targets_a, targets_b))
            else:
                train_targets = targets

            if self.args.multiple_GPUs:
                _, loss = self.model(images, train_targets, mixup_lambda)
            else:
                _, loss = self.model(images.cuda(), train_targets.cuda(), mixup_lambda)

            loss = loss.mean()
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()

            self.model.eval()
            with torch.no_grad():
                correct, labeled, inter, union = self.eval_batch(
                    self.model, images, targets
                )

            total_correct += correct
            total_label += labeled
            total_inter += inter
            total_union += union
            pixAcc = 1.0 * total_correct / (np.spacing(1) + total_label)
            IoU = 1.0 * total_inter / (np.spacing(1) + total_union)
            mIoU = IoU.mean()

            tbar.set_description(
                "Train loss: %.3f   mIoU: %.3f" % (train_loss / (i + 1), mIoU)
            )

            iteration = epoch * len(self.trainloader) + i + 1
            self.writer.add_scalar("xtras/epoch", epoch + 1, iteration)
            self.writer.add_scalar("xtras/learning_rate", lr, iteration)
            if (i + 1) % 100 == 0:
                self.writer.add_scalar(
                    "train_step/loss", train_loss / (i + 1), iteration
                )
                self.writer.add_scalar("train_step/mIoU", mIoU, iteration)
                self.writer.add_scalar("train_step/pixAcc", pixAcc, iteration)

            # LookAhead optimizer update step with alpha
            if self.args.lookAhead_steps is None:
                pass
            elif (
                self.args.lookAhead_steps != 0
                and (i % self.args.lookAhead_steps) == self.args.lookAhead_steps - 1
            ):
                curr_state = self.model.module.state_dict()
                for key in curr_state.keys():
                    # factored out alpha to do the update in-place for speed
                    curr_state[key] += (1.0 / self.lookAhead_alpha - 1) * prev_state[
                        key
                    ]
                    curr_state[key] *= self.lookAhead_alpha

                prev_state = copy.deepcopy(curr_state)

        if self.args.save_interval > 0:
            if (epoch + 1) % self.args.save_interval == 0:
                # save checkpoint every epoch
                model_to_save = (
                    self.model.module.model
                    if self.args.multiple_GPUs
                    else self.model.model
                )
                is_best = False
                save_checkpoint(
                    {
                        "epoch": epoch + 1,
                        "state_dict": model_to_save.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                        "best_pred": self.best_pred,
                    },
                    self.directory,
                    is_best,
                    filename="checkpoint_%03d.pth.tar" % (epoch + 1),
                )

        # print(f"adding train {epoch}")
        self.writer.add_scalar("train_epoch/loss", train_loss / (i + 1), epoch + 1)
        self.writer.add_scalar("train_epoch/mIoU", mIoU, epoch + 1)
        self.writer.add_scalar("train_epoch/pixAcc", pixAcc, epoch + 1)
        if self.args.tb_images:
            for ii in range(4):
                self.writer.add_image(
                    f"train/image{ii}", self.inverse_transform(images[ii]), iteration
                )

        # Fast test during the training

    def eval_batch(self, model, images, targets, return_loss=False):
        if self.args.multiple_GPUs:
            outputs, loss = model(images, targets)
        else:
            outputs, loss = model(images.cuda(), targets.cuda())

        loss = loss.mean()
        correct, labeled = utils.batch_pix_accuracy(outputs.data, targets)
        inter, union = utils.batch_intersection_union(
            outputs.data, targets, self.nclass
        )
        if return_loss:
            return correct, labeled, inter, union, loss
        else:
            return correct, labeled, inter, union

    def validation(self, epoch):
        is_best = False
        self.model.eval()
        total_inter, total_union, total_correct, total_label, total_loss = 0, 0, 0, 0, 0
        tbar = tqdm(self.valloader, ascii=True)

        for i, (images, targets) in enumerate(tbar):
            with torch.no_grad():
                correct, labeled, inter, union, loss = self.eval_batch(
                    self.model, images, targets, return_loss=True
                )

            total_correct += correct
            total_label += labeled
            total_inter += inter
            total_union += union
            total_loss += loss
            pixAcc = 1.0 * total_correct / (np.spacing(1) + total_label)
            IoU = 1.0 * total_inter / (np.spacing(1) + total_union)
            mIoU = IoU.mean()

            tbar.set_description(
                "Val   loss: %.3f   mIoU: %.3f" % (total_loss / (i + 1), mIoU)
            )

        # print(f"adding validation {epoch}")
        self.writer.add_scalar("val_epoch/mIoU", mIoU, epoch + 1)
        self.writer.add_scalar("val_epoch/pixAcc", pixAcc, epoch + 1)
        self.writer.add_scalar("val_epoch/loss", (total_loss) / (i + 1), epoch + 1)
        if self.args.tb_images:
            for ii in range(4):
                self.writer.add_image(
                    f"val/image{ii}", self.inverse_transform(images[ii]), epoch + 1
                )
        self.writer.flush()

        new_pred = (pixAcc + mIoU) / 2
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
            model_to_save = (
                self.model.module.model if self.args.multiple_GPUs else self.model.model
            )
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": model_to_save.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "best_pred": self.best_pred,
                },
                self.directory,
                is_best,
            )


def main():
    args = Options().parse()
    args.custom_dict = json.loads(args.custom_dict)
    print(args)
    torch.manual_seed(args.seed)

    checkname = (
        args.checkname + "_" + str(args.backbone) + "_" + "%04d" % args.batch_size
    )
    print(f"checkname is {checkname}")
    directory = os.path.join(
        args.save_dir, "%s/%s/%s/" % (args.dataset, args.model, checkname)
    )

    trainer = Trainer(args, directory)
    print("Starting Epoch:", args.start_epoch)
    print("Total Epochs:", args.epochs)

    if args.clr_finder:
        trainer.findlr()
        return

    with EnergyMeter(writer=trainer.writer, dir=directory) if args.em else ExitStack():
        for epoch in range(args.start_epoch, args.epochs):
            trainer.training(epoch)
            if not args.no_val:
                trainer.validation(epoch)


if __name__ == "__main__":
    main()
