#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import random
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

from classifier_utils import find_classes
from conf import settings
from dataset import CustomData


def class_plot(data, inv_normalizen_figures=12):
    n_row = int(12 / 3)
    fig, axes = plt.subplots(figsize=(14, 10), nrows=n_row, ncols=3)
    i = 0
    for ax in axes.flatten():
        a = random.randint(0, len(data))  # list(range(58,len(data)))
        (image, label) = data[a]  # data[a[i]]
        label = int(label)
        # l = encoder[label]
        # image = inv_normalize(image)
        image = image.numpy().transpose(1, 2, 0)
        ax.set_title(label)
        ax.axis("off")
        plt.show()
        i = i + 1


def get_training_dataloader(
    train_dir,
    batch_size=16,
    num_workers=2,
    shuffle=True,
    dataset_type=None,
    img_size=44,
):
    """ return training dataloader
    Args:
        mean: mean of cifar100 training dataset
        std: std of cifar100 training dataset
        path: path to cifar100 training python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: train_data_loader:torch dataloader object
    """

    if dataset_type == "cifar":
        # cifar100_training = CIFAR100Train(path, transform=transform_train)
        mean = settings.CIFAR100_TRAIN_MEAN
        std = settings.CIFAR100_TRAIN_STD
        transform_train = transforms.Compose(
            [
                # transforms.ToPILImage(),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
                # transforms.SubtractMeans(mean)
            ]
        )
        cifar100_training = torchvision.datasets.CIFAR100(
            root=train_dir, train=True, download=True, transform=transform_train
        )
        training_loader = DataLoader(
            cifar100_training,
            shuffle=shuffle,
            num_workers=num_workers,
            batch_size=batch_size,
        )
        class_plot(cifar100_training, 12)

    else:
        mean = settings.IMAGENET_MEAN
        std = settings.IMAGENET_STD
        transform_train = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.RandomPerspective(),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
            ]
        )
        _, class_to_idx = find_classes(dataset_type, train_dir)
        train_data = CustomData(
            train_dir,
            class_to_idx,
            transform=transform_train,
            dataset_type=dataset_type,
        )
        training_loader = DataLoader(
            train_data, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size
        )

    return training_loader


def get_test_dataloader(
    test_dir,
    train_dir=[],
    batch_size=16,
    num_workers=2,
    shuffle=True,
    dataset_type=None,
    img_size=44,
):
    """ return training dataloader
    Args:
        mean: mean of cifar100 test dataset
        std: std of cifar100 test dataset
        path: path to cifar100 test python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: cifar100_test_loader:torch dataloader object
    """

    if dataset_type == "cifar":
        # cifar100_test = CIFAR100Test(path, transform=transform_test)

        mean = settings.CIFAR100_TRAIN_MEAN
        std = settings.CIFAR100_TRAIN_STD
        transform_test = transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
                # transforms.SubtractMeans(mean)
            ]
        )
        cifar100_test = torchvision.datasets.CIFAR100(
            root=test_dir, train=False, download=True, transform=transform_test
        )
        test_loader = DataLoader(
            cifar100_test,
            shuffle=shuffle,
            num_workers=num_workers,
            batch_size=batch_size,
        )
    else:
        mean = settings.IMAGENET_MEAN
        std = settings.IMAGENET_STD
        transform_test = transforms.Compose(
            [transforms.Resize((img_size, img_size)), transforms.ToTensor(),]
        )
        _, class_to_idx = find_classes(dataset_type, train_dir)
        test_data = CustomData(
            test_dir, class_to_idx, transform=transform_test, dataset_type=dataset_type
        )
        test_loader = DataLoader(
            test_data, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size
        )

    return test_loader
