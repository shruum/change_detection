from od.modeling.anchors.prior_box import PriorBox
from od.data.transforms.transforms import *
from od.data.transforms.target_transform import (
    SSDTargetTransform,
    YoloTargetTransform,
    CenterNetHeadTransform,
    ThunderNetTargetTransform,
)

"""
First we list out the transforms for each head
1) ThunderNet
2) SSD
3) YOLO v2 & v3
4) CenterNet
"""


def transform_ThunderNet(cfg, is_train=True):
    """
    Data transforms for ThunderNetHead
    :param cfg: config file
    :param is_train: parameter to differentiate between train and test mode
    :return: transformed data
    """
    if is_train:
        transform = [
            ConvertFromInts(),
            PhotometricDistort(),
            Expand(cfg.INPUT.PIXEL_MEAN),
            RandomSampleCrop(),
            RandomMirror(),
            ResizeImageBoxes(cfg, True),
            SubtractMeans(cfg.INPUT.PIXEL_MEAN),
            ToTensor(),
        ]
    else:
        transform = [
            ResizeImageBoxes(cfg, False),
            SubtractMeans(cfg.INPUT.PIXEL_MEAN),
            ToTensor(),
        ]
    transform = Compose(transform)
    return transform


def transform_SSD(cfg, is_train=True):
    """
    Data transforms for SSD box head
    :param cfg: config file
    :param is_train: parameter to differentiate between train and test mode
    :return: transformed data
    """
    if is_train:
        transform = [
            ConvertFromInts(),
            PhotometricDistort(),
            Expand(cfg.INPUT.PIXEL_MEAN),
            RandomSampleCrop(),
            RandomMirror(),
            ToPercentCoords(),
            Resize(cfg.INPUT.IMAGE_SIZE),
            SubtractMeans(cfg.INPUT.PIXEL_MEAN),
            ToTensor(),
        ]
    else:
        transform = [
            Resize(cfg.INPUT.IMAGE_SIZE),
            SubtractMeans(cfg.INPUT.PIXEL_MEAN),
            ToTensor(),
        ]
    transform = Compose(transform)
    return transform


def transform_Yolo(cfg, is_train=True):
    """
    Data transforms for YOLO v2 and v3
    :param cfg: config file
    :param is_train: parameter to differentiate between train and test mode
    :return: transformed data
    """
    if is_train:
        transform = [
            RandomFlip(),
            YoloCrop(),
            ToPercentCoords(),
            Resize(cfg.INPUT.IMAGE_SIZE),
            SubtractMeans(cfg.INPUT.PIXEL_MEAN),
            ToTensor(),
        ]
    else:
        transform = [
            Resize(cfg.INPUT.IMAGE_SIZE),
            SubtractMeans(cfg.INPUT.PIXEL_MEAN),
            ToTensor(),
        ]
    transform = Compose(transform)
    return transform


def transform_CenterNet(cfg, is_train=True):
    """
    Data transforms for CenterNetHead
    :param cfg: config file
    :param is_train: parameter to differentiate between train and test mode
    :return: transformed data
    """
    if is_train:
        transform = [
            ConvertFromInts(),
            PhotometricDistort(),
            RandomSampleCrop(),
            RandomMirror(),
            ToPercentCoords(),
            Resize(cfg.INPUT.IMAGE_SIZE),
            SubtractMeans(cfg.INPUT.PIXEL_MEAN),
            ToTensor(),
        ]
    else:
        transform = [
            Resize(cfg.INPUT.IMAGE_SIZE),
            SubtractMeans(cfg.INPUT.PIXEL_MEAN),
            ToTensor(),
        ]
    transform = Compose(transform)
    return transform


"""
Now, we write the main controller functions to call Head-Specific transforms 
"""


def build_transforms(cfg, is_train=True):
    """
    Image transforms (for all heads)
    :param cfg: config file
    :param is_train: train or test mode
    :return: return the transformations for the calling head
    """
    if "SSD" in cfg.MODEL.HEAD.NAME:
        return transform_SSD(cfg, is_train)
    elif "ThunderNet" in cfg.MODEL.HEAD.NAME:
        return transform_ThunderNet(cfg, is_train)
    elif "Yolov2Head" in cfg.MODEL.HEAD.NAME or "Yolov3Head" in cfg.MODEL.HEAD.NAME:
        return transform_Yolo(cfg, is_train)
    elif "CenterNetHead" in cfg.MODEL.HEAD.NAME:
        return transform_CenterNet(cfg, is_train)
    else:
        raise NotImplementedError(
            "Transformation for detection head {} not implemented.".format(
                cfg.MODEL.HEAD.NAME
            )
        )


def build_target_transform(cfg):
    """
    Target transforms (for all heads) - ground truth boxes and labels or heatmaps
    :param cfg: config file
    :return: return the transformations for the calling head
    """
    if "SSD" in cfg.MODEL.HEAD.NAME:
        return SSDTargetTransform(
            PriorBox(cfg)(),
            cfg.MODEL.CENTER_VARIANCE,
            cfg.MODEL.SIZE_VARIANCE,
            cfg.MODEL.THRESHOLD,
        )
    elif "ThunderNet" in cfg.MODEL.HEAD.NAME:
        return ThunderNetTargetTransform(cfg.INPUT.MAX_NUM_GT_BOXES)
    elif "Yolov2Head" in cfg.MODEL.HEAD.NAME or "Yolov3Head" in cfg.MODEL.HEAD.NAME:
        return YoloTargetTransform(cfg.INPUT.MAX_NUM_GT_BOXES)
    elif "CenterNetHead" in cfg.MODEL.HEAD.NAME:
        return CenterNetHeadTransform(cfg)
    else:
        raise NotImplementedError(
            "Target transformation for detection head {} not implemented.".format(
                cfg.MODEL.HEAD.NAME
            )
        )
