# train.py
# !/usr/bin/env	python3

""" train network using pytorch

author baiyu
"""

import argparse
from contextlib import ExitStack
import numpy as np
import os
from random import uniform
from tensorboardX import SummaryWriter
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim


from conf import settings
from dataloader import get_training_dataloader, get_test_dataloader
from energy_meter import EnergyMeter
from classifier_utils import (
    get_network,
    get_optimizer,
    WarmUpLR,
    count_parameters,
)

total = 0
train_total = 0
val_total = 0


def train(epoch):
    net.train()

    global train_total
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    loss_list = []
    for batch_index, (images, labels) in enumerate(training_loader):
        if epoch <= args.warm:
            warmup_scheduler.step()

        images = Variable(images)
        labels = Variable(labels)

        labels = labels.cuda()
        images = images.cuda()

        optimizer.zero_grad()

        start.record()

        outputs = net(images)

        end.record()
        torch.cuda.synchronize()
        time_elapsed = start.elapsed_time(end)
        train_total += time_elapsed

        loss = loss_function(outputs, labels)
        if args.random_gradient:
            loss *= uniform(0, 1)
        loss.backward()

        optimizer.step()

        n_iter = (epoch - 1) * len(training_loader) + batch_index + 1

        last_layer = list(net.children())[-1]
        for name, para in last_layer.named_parameters():
            if "weight" in name:
                writer.add_scalar(
                    "LastLayerGradients/grad_norm2_weights", para.grad.norm(), n_iter
                )
            if "bias" in name:
                writer.add_scalar(
                    "LastLayerGradients/grad_norm2_bias", para.grad.norm(), n_iter
                )

        loss_list.append(loss.item())
        if batch_index % 100 == 0:
            loss_mean = np.mean(np.array(loss_list))
            loss_list = []
            print(
                "Training Epoch: {epoch} [{batch_index}/{total_num_batches}]\tLoss: {:0.4f}\tLR: {:0.6f}".format(
                    loss_mean,
                    optimizer.param_groups[0]["lr"],
                    epoch=epoch,
                    batch_index=batch_index,
                    total_num_batches=len(training_loader),
                ),
                end="\r",
                flush=True,
            )
        # update training loss for each iteration
        writer.add_scalar("Train/loss", loss.item(), n_iter)

    for name, param in net.named_parameters():
        layer, attr = os.path.splitext(name)
        attr = attr[1:]
        writer.add_histogram("{}/{}".format(layer, attr), param, epoch)


def eval_training(epoch):
    net.eval()
    global val_total
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    test_loss = 0.0  # cost function error
    correct = 0.0

    for (images, labels) in test_loader:
        images = Variable(images)
        labels = Variable(labels)

        images = images.cuda()
        labels = labels.cuda()

        start.record()

        outputs = net(images)

        end.record()
        torch.cuda.synchronize()
        time_elapsed = start.elapsed_time(end)
        val_total += time_elapsed

        loss = loss_function(outputs, labels)
        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

    print(
        "Test set: Average loss: {:.4f}, Accuracy: {:.4f}".format(
            test_loss / len(test_loader.dataset),
            correct.float() / len(test_loader.dataset),
        )
    )
    print()

    # add informations to tensorboard
    writer.add_scalar("Test/Average loss", test_loss / len(test_loader.dataset), epoch)
    writer.add_scalar(
        "Test/Accuracy", correct.float() / len(test_loader.dataset), epoch
    )

    return correct.float() / len(test_loader.dataset)


def save_model(args, net, optimizer, train_scheduler, epoch, save_type="regular"):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": train_scheduler.state_dict(),
        },
        checkpoint_path.format(net=args.net, epoch=epoch, type=save_type),
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("train_dir", type=str, help="Path to training set")
    parser.add_argument(
        "test_dir",
        default="",
        type=str,
        help="Path to test set",
    )
    parser.add_argument("net", type=str, help="Name of network (see 'get_network' in 'classifier_utils.py')")
    parser.add_argument(
        "--batch_size", "-b", type=int, default=128, help="batch size for dataloader"
    )
    parser.add_argument("--checkname", type=str, help="experiment name")
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="cifar",
        help="specify the dataset type. Default is CIFAR",
    )
    parser.add_argument("--energy_meter", "-em", default=True, help="METER ENERGY")
    parser.add_argument("--epochs", type=int, default=120, help="Number of epochs")
    parser.add_argument("--gpu", type=bool, default=True, help="use gpu or not")
    parser.add_argument(
        "--img_size", type=int, default=32, help="Size to resize images"
    )
    parser.add_argument(
        "--learning_rate", "-lr", type=float, default=0.1, help="initial learning rate"
    )
    parser.add_argument("--loss", default="cross_entropy", help="path where to save")
    parser.add_argument(
        "--milestones", type=int, action="append", default=[], help="Number of epochs"
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=100,
        help="Number of classes. Default is 100 for CIFAR",
    )
    parser.add_argument(
        "--num_workers",
        "-w",
        type=int,
        default=8,
        help="number of workers for dataloader",
    )
    parser.add_argument("--optimizer", type=str, default="sgd", help="choose optimizer")
    parser.add_argument("--output_dir", default="./runs", help="path where to save")
    parser.add_argument(
        "--random_gradient", action="store_true", help="apply random gradient"
    )
    parser.add_argument(
        "--shuffle", "-s", type=bool, default=True, help="whether shuffle the dataset"
    )
    parser.add_argument("--warm", type=int, default=1, help="warm up training phase")
    args = parser.parse_args()

    # data preprocessing:
    NUM_CLASSES = args.num_classes
    settings.NUM_CLASSES = NUM_CLASSES
    settings.EPOCH = args.epochs
    if args.milestones:
        settings.MILESTONES = args.milestones

    training_loader = get_training_dataloader(
        train_dir=args.train_dir,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        dataset_type=args.dataset_type,
        img_size=args.img_size,
    )
    test_loader = get_test_dataloader(
        test_dir=args.test_dir,
        train_dir=args.train_dir,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        dataset_type=args.dataset_type,
        img_size=args.img_size,
    )

    if args.loss == "cross_entropy":
        loss_function = nn.CrossEntropyLoss()
    else:
        raise ValueError

    net = get_network(args.net, args.num_classes, use_gpu=args.gpu)
    optimizer = get_optimizer(
        args.optimizer,
        params=net.parameters(),
        lr=args.learning_rate,
        momentum=0.9,
        weight_decay=5e-4,
    )
    init_epoch = 1
    train_scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=settings.MILESTONES, gamma=0.1
    )  # learning rate decay

    iter_per_epoch = len(training_loader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)

    log_path = os.path.join(args.output_dir, args.net, args.checkname)
    writer = SummaryWriter(log_dir=os.path.join(log_path, "summary"))
    input_tensor = torch.Tensor(12, 3, 32, 32).cuda()
    checkpoint_path = os.path.join(log_path, "{net}-{epoch}-{type}.pth")

    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        net.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        train_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        init_epoch = checkpoint["epoch"]

    os.makedirs(log_path, exist_ok=True)

    print("################################################")
    print("## Number of Parameters")
    print("##", count_parameters(model=net))
    print("################################################")

    total_start = torch.cuda.Event(enable_timing=True)
    total_end = torch.cuda.Event(enable_timing=True)
    total_start.record()

    with EnergyMeter(writer=writer, dir=log_path) if args.energy_meter else ExitStack():
        # with torch.autograd.profiler.profile(use_cuda=True) as prof:
        best_acc = 0.0
        best_epoch = -1
        for epoch in range(init_epoch, settings.EPOCH):

            if epoch > args.warm:
                train_scheduler.step(epoch)

            train(epoch)
            if epoch % settings.SAVE_EPOCH == 0:
                acc = eval_training(epoch)
                save_model(
                    args, net, optimizer, train_scheduler, epoch, save_type="regular"
                )

            # start to save best performance model after learning rate decay to 0.01
            if epoch > settings.MILESTONES[1] and best_acc < acc:
                save_model(
                    args, net, optimizer, train_scheduler, epoch, save_type="best"
                )
                best_acc = acc
                best_epoch = epoch
                continue

        writer.close()

    total_end.record()
    torch.cuda.synchronize()
    total_time = total_start.elapsed_time(total_end)

    print("total train(Gpu only)", str(train_total))
    print("total val(Gpu only)", str(val_total))
    print("total train+val(Gpu only)", str(train_total + val_total))
    print("total train+val(cpu+Gpu)", str(total_time))
    print("#########################################")
    print("Best Acc: ", best_acc)
    print("Best Epoch: ", best_epoch)
