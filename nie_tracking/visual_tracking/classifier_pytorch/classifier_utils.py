""" helper function

author baiyu
"""

import json
import numpy
import os
import sys
from torch import optim
from torch.optim.lr_scheduler import _LRScheduler

from classes import get_class_dict

CURRENT_DIR = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(CURRENT_DIR, "modules"))
from radam import RAdam


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_optimizer(name, **kwargs):
    if name == "sgd":
        optimizer = optim.SGD(**kwargs)
    elif name == "radam":
        kwargs.pop("momentum", None)
        optimizer = RAdam(**kwargs)
    else:
        raise ValueError("Unkown Optimizer")
    return optimizer


def get_network(net, num_classes=100, use_gpu=True):
    """ return given network
    """
    if net == "vgg16":
        from models.vgg import vgg16_bn

        net = vgg16_bn()
    elif net == "vgg13":
        from models.vgg import vgg13_bn

        net = vgg13_bn()
    elif net == "vgg11":
        from models.vgg import vgg11_bn

        net = vgg11_bn()
    elif net == "vgg19":
        from models.vgg import vgg19_bn

        net = vgg19_bn()
    elif net == "densenet121":
        from models.densenet import densenet121

        net = densenet121()
    elif net == "densenet161":
        from models.densenet import densenet161

        net = densenet161()
    elif net == "densenet169":
        from models.densenet import densenet169

        net = densenet169()
    elif net == "densenet201":
        from models.densenet import densenet201

        net = densenet201()
    elif net == "googlenet":
        from models.googlenet import googlenet

        net = googlenet()
    elif net == "inceptionv3":
        from models.inceptionv3 import inceptionv3

        net = inceptionv3()
    elif net == "inceptionv4":
        from models.inceptionv4 import inceptionv4

        net = inceptionv4()
    elif net == "inceptionresnetv2":
        from models.inceptionv4 import inception_resnet_v2

        net = inception_resnet_v2()
    elif net == "xception":
        from models.xception import xception

        net = xception()
    elif net == "resnet18":
        from models.resnet import resnet18

        net = resnet18(num_classes)
    elif net == "resnet34":
        from models.resnet import resnet34

        net = resnet34(num_classes)
    elif net == "resnet50":
        from models.resnet import resnet50

        net = resnet50(num_classes)
    elif net == "resnet101":
        from models.resnet import resnet101

        net = resnet101(num_classes)
    elif net == "resnet152":
        from models.resnet import resnet152

        net = resnet152(num_classes)
    elif net == "preactresnet18":
        from models.preactresnet import preactresnet18

        net = preactresnet18()
    elif net == "preactresnet34":
        from models.preactresnet import preactresnet34

        net = preactresnet34()
    elif net == "preactresnet50":
        from models.preactresnet import preactresnet50

        net = preactresnet50()
    elif net == "preactresnet101":
        from models.preactresnet import preactresnet101

        net = preactresnet101()
    elif net == "preactresnet152":
        from models.preactresnet import preactresnet152

        net = preactresnet152()
    elif net == "resnext50":
        from models.resnext import resnext50

        net = resnext50()
    elif net == "resnext101":
        from models.resnext import resnext101

        net = resnext101()
    elif net == "resnext152":
        from models.resnext import resnext152

        net = resnext152()
    elif net == "shufflenet":
        from models.shufflenet import shufflenet

        net = shufflenet()
    elif net == "shufflenetv2":
        from models.shufflenetv2 import shufflenetv2

        net = shufflenetv2()
    elif net == "squeezenet":
        from models.squeezenet import squeezenet

        net = squeezenet()
    elif net == "mobilenet":
        from models.mobilenet import mobilenet

        net = mobilenet()
    elif net == "mobilenetv2":
        from models.mobilenetv2 import mobilenetv2

        net = mobilenetv2()
    elif net == "nasnet":
        from models.nasnet import nasnet

        net = nasnet()
    elif net == "attention56":
        from models.attention import attention56

        net = attention56()
    elif net == "attention92":
        from models.attention import attention92

        net = attention92()
    elif net == "seresnet18":
        from models.senet import seresnet18

        net = seresnet18()
    elif net == "seresnet34":
        from models.senet import seresnet34

        net = seresnet34()
    elif net == "seresnet50":
        from models.senet import seresnet50

        net = seresnet50()
    elif net == "seresnet101":
        from models.senet import seresnet101

        net = seresnet101()
    elif net == "seresnet152":
        from models.senet import seresnet152

        net = seresnet152()
    elif net == "coord_resnet50":
        from models.coord_resnet import cc_resnet50

        net = cc_resnet50()
    elif net == "switch_resnet50":
        from models.switch_resnet import resnet50

        net = resnet50()
    elif net == "sp_resnet50":
        from models.sp_resnet import resnet50

        net = resnet50()
    else:
        print("the network name you have entered is not supported yet")
        sys.exit()

    if use_gpu:
        net = net.cuda()

    return net


def compute_mean_std(cifar100_dataset):
    """compute the mean and std of cifar100 dataset
    Args:
        cifar100_training_dataset or cifar100_test_dataset
        witch derived from class torch.utils.data

    Returns:
        a tuple contains mean, std value of entire dataset
    """

    data_r = numpy.dstack(
        [cifar100_dataset[i][1][:, :, 0] for i in range(len(cifar100_dataset))]
    )
    data_g = numpy.dstack(
        [cifar100_dataset[i][1][:, :, 1] for i in range(len(cifar100_dataset))]
    )
    data_b = numpy.dstack(
        [cifar100_dataset[i][1][:, :, 2] for i in range(len(cifar100_dataset))]
    )
    mean = numpy.mean(data_r), numpy.mean(data_g), numpy.mean(data_b)
    std = numpy.std(data_r), numpy.std(data_g), numpy.std(data_b)

    return mean, std


def compute_mean_std1(dataloader):
    mean = 0.0
    std = 0.0
    nb_samples = len(dataloader.dataset)
    for data, _ in dataloader:
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
    mean /= nb_samples
    std /= nb_samples

    return mean, std


def find_classes(dataset_type=None, data_dir="."):
    class_to_idx = get_class_dict(dataset_type)
    if len(class_to_idx) == 0:
        filepath = os.path.join(data_dir, "class_to_idx.json")
        if os.path.isfile(filepath):
            with open(filepath, "r") as file:
                class_to_idx = json.load(file)
        else:
            classes = [d.name for d in os.scandir(data_dir) if d.is_dir()]
            classes.sort()
            class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
            with open(filepath, "w") as file:
                json.dump(class_to_idx, file)
    classes = []
    for key in class_to_idx:
        classes.append(key)
    return classes, class_to_idx


class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """

    def __init__(self, optimizer, total_iters, last_epoch=-1):
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [
            base_lr * self.last_epoch / (self.total_iters + 1e-8)
            for base_lr in self.base_lrs
        ]
