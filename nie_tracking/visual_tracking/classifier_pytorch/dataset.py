""" train and test dataset

author baiyu
"""

import numpy
import os
import pickle
from PIL import Image
from torch.utils.data import Dataset


class CIFAR100Train(Dataset):
    """cifar100 test dataset, derived from
    torch.utils.data.DataSet
    """

    def __init__(self, path, transform=None):
        # if transform is given, we transoform data using
        with open(os.path.join(path, "train"), "rb") as cifar100:
            self.data = pickle.load(cifar100, encoding="bytes")
        self.transform = transform

    def __len__(self):
        return len(self.data["fine_labels".encode()])

    def __getitem__(self, index):
        label = self.data["fine_labels".encode()][index]
        r = self.data["data".encode()][index, :1024].reshape(32, 32)
        g = self.data["data".encode()][index, 1024:2048].reshape(32, 32)
        b = self.data["data".encode()][index, 2048:].reshape(32, 32)
        image = numpy.dstack((r, g, b))

        if self.transform:
            image = self.transform(image)
        return label, image


class CIFAR100Test(Dataset):
    """cifar100 test dataset, derived from
    torch.utils.data.DataSet
    """

    def __init__(self, path, transform=None):
        with open(os.path.join(path, "test"), "rb") as cifar100:
            self.data = pickle.load(cifar100, encoding="bytes")
        self.transform = transform

    def __len__(self):
        return len(self.data["data".encode()])

    def __getitem__(self, index):
        label = self.data["fine_labels".encode()][index]
        r = self.data["data".encode()][index, :1024].reshape(32, 32)
        g = self.data["data".encode()][index, 1024:2048].reshape(32, 32)
        b = self.data["data".encode()][index, 2048:].reshape(32, 32)
        image = numpy.dstack((r, g, b))

        if self.transform:
            image = self.transform(image)
        return label, image


def loader(path):
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


IMG_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".ppm",
    ".bmp",
    ".pgm",
    ".tif",
    ".tiff",
    ".webp",
)


def has_file_allowed_extension(filename, extensions):
    return filename.lower().endswith(extensions)


def make_dataset(
    directory, class_to_idx, extensions=None, is_valid_file=None,
):
    instances = []
    directory = os.path.expanduser(directory)
    both_none = extensions is None and is_valid_file is None
    both_something = extensions is not None and is_valid_file is not None
    if both_none or both_something:
        raise ValueError(
            "Both extensions and is_valid_file cannot be None or not None at the same time"
        )
    if extensions is not None:

        def is_valid_file(x):
            return has_file_allowed_extension(x, extensions)

    for target_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[target_class]
        target_dir = os.path.join(directory, target_class)
        if not os.path.isdir(target_dir):
            continue
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            samples = sorted(fnames)
            paths = [os.path.join(root, sample) for sample in samples]
            for path in paths:
                if is_valid_file(path):
                    item = path, class_index
                    instances.append(item)
    return instances


class CustomData(Dataset):
    """cifar100 test dataset, derived from
    torch.utils.data.DataSet
    """

    def __init__(
        self, path, class_to_idx, transform=None, is_valid_file=None, dataset_type=None
    ):
        self.samples = make_dataset(
            path,
            class_to_idx,
            IMG_EXTENSIONS if is_valid_file is None else None,
            is_valid_file=is_valid_file,
        )
        self.transform = transform
        self.loader = loader
        self.dataset_type = dataset_type

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        path, label = self.samples[index]
        img = self.loader(path)

        if self.transform:
            image = self.transform(img)

        return image, label
