# ------------------------------------------------------------------------------
# Portions of this code are from
# CornerNet (https://github.com/princeton-vl/CornerNet)
# Copyright (c) 2019, NavInfo Europe
# NOTE:Implementation of CenterNet Head in Detection as a service Framework
# ------------------------------------------------------------------------------
# trying to merge this

import torch
import torch.nn as nn
from od.modeling.head.centernet.utils import (
    _gather_feat,
    _tranpose_and_gather_feat,
    _tranpose_and_gather_feat_onnx,
)
from od.structures.container import Container
from onnxsupport import CenternetHelperOperations, CenternetHelper


def _nms_onnx(heat, kernel=3):
    return CenternetHelper()(
        heat, heat, CenternetHelperOperations.CENTERNET_NMS.value, kernel
    )


def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep


def _left_aggregate(heat):
    """
    heat: batchsize x channels x h x w
    """
    shape = heat.shape
    heat = heat.reshape(-1, heat.shape[3])
    heat = heat.transpose(1, 0).contiguous()
    ret = heat.clone()
    for i in range(1, heat.shape[0]):
        inds = heat[i] >= heat[i - 1]
        ret[i] += ret[i - 1] * inds.float()
    return (ret - heat).transpose(1, 0).reshape(shape)


def _right_aggregate(heat):
    """
    heat: batchsize x channels x h x w
    """
    shape = heat.shape
    heat = heat.reshape(-1, heat.shape[3])
    heat = heat.transpose(1, 0).contiguous()
    ret = heat.clone()
    for i in range(heat.shape[0] - 2, -1, -1):
        inds = heat[i] >= heat[i + 1]
        ret[i] += ret[i + 1] * inds.float()
    return (ret - heat).transpose(1, 0).reshape(shape)


def _top_aggregate(heat):
    """
    heat: batchsize x channels x h x w
    """
    heat = heat.transpose(3, 2)
    shape = heat.shape
    heat = heat.reshape(-1, heat.shape[3])
    heat = heat.transpose(1, 0).contiguous()
    ret = heat.clone()
    for i in range(1, heat.shape[0]):
        inds = heat[i] >= heat[i - 1]
        ret[i] += ret[i - 1] * inds.float()
    return (ret - heat).transpose(1, 0).reshape(shape).transpose(3, 2)


def _bottom_aggregate(heat):
    """
    heat: batchsize x channels x h x w
    """
    heat = heat.transpose(3, 2)
    shape = heat.shape
    heat = heat.reshape(-1, heat.shape[3])
    heat = heat.transpose(1, 0).contiguous()
    ret = heat.clone()
    for i in range(heat.shape[0] - 2, -1, -1):
        inds = heat[i] >= heat[i + 1]
        ret[i] += ret[i + 1] * inds.float()
    return (ret - heat).transpose(1, 0).reshape(shape).transpose(3, 2)


def _h_aggregate(heat, aggr_weight=0.1):
    return (
        aggr_weight * _left_aggregate(heat)
        + aggr_weight * _right_aggregate(heat)
        + heat
    )


def _v_aggregate(heat, aggr_weight=0.1):
    return (
        aggr_weight * _top_aggregate(heat)
        + aggr_weight * _bottom_aggregate(heat)
        + heat
    )


def _topk_channel(scores, K=40):
    batch, cat, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()

    return topk_scores, topk_inds, topk_ys, topk_xs


def _topk_onnx(scores, K=40):
    result = CenternetHelper()(scores, scores, CenternetHelperOperations.TOPK.value, K)
    # Cannot return multiple tensors from tensorrt so it is best to concatenate them
    # Parse results
    topk_score = result[0:1, :]
    topk_inds = result[1:2, :]
    topk_clses = result[2:3, :]
    topk_ys = result[3:4, :]
    topk_xs = result[4:5, :]

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


def _topk(scores, K=40):
    batch, cat, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()

    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_clses = (topk_ind / K).int()
    topk_inds = _gather_feat(topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


def centernet_decode(
    heat, wh, image_size, onxx_export=False, reg=None, cat_spec_wh=False, K=100
):
    if onxx_export:
        batch = int(heat.size(0))
        cat = int(heat.size(1))
        height = int(heat.size(2))
        width = int(heat.size(3))
        heat = _nms_onnx(heat)
    else:
        batch, cat, height, width = heat.size()
        # heat = torch.sigmoid(heat)
        # perform nms on heatmaps
        heat = _nms(heat)

    if onxx_export:
        scores, inds, clses, ys, xs = _topk_onnx(heat, K=K)
    else:
        scores, inds, clses, ys, xs = _topk(heat, K=K)
    if reg is not None:
        if onxx_export:
            reg = _tranpose_and_gather_feat_onnx(reg, inds)
            reg = reg.reshape(batch, -1, 2)
            xs = xs.reshape(batch, -1, 1)
            xs = xs + reg[:, :, 0:1]
            ys = ys.reshape(batch, -1, 1)
            ys = ys + reg[:, :, 1:2]
        else:
            reg = _tranpose_and_gather_feat(reg, inds)
            reg = reg.view(batch, K, 2)
            xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
            ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
    else:
        xs = xs.view(batch, K, 1) + 0.5
        ys = ys.view(batch, K, 1) + 0.5
    if onxx_export:
        wh = _tranpose_and_gather_feat_onnx(wh, inds)
    else:
        wh = _tranpose_and_gather_feat(wh, inds)
    if cat_spec_wh:
        wh = wh.view(batch, K, cat, 2)
        clses_ind = clses.view(batch, K, 1, 1).expand(batch, K, 1, 2).long()
        wh = wh.gather(2, clses_ind).view(batch, K, 2)
    else:
        if onxx_export:
            wh = wh.reshape(batch, -1, 2)
        else:
            wh = wh.view(batch, K, 2)

    if onxx_export:
        clses = clses.reshape(batch, -1, 1)
        scores = scores.reshape(batch, -1, 1)
    else:
        clses = clses.view(batch, K, 1).float()
        scores = scores.view(batch, K, 1)

    bboxes = torch.cat(
        [
            xs - wh[..., 0:1] / 2,
            ys - wh[..., 1:2] / 2,
            xs + wh[..., 0:1] / 2,
            ys + wh[..., 1:2] / 2,
        ],
        dim=2,
    )

    clses = clses.squeeze(dim=2)
    scores = scores.squeeze(dim=2)

    """
    Comment: The output is now converted to a format common for all the heads. This reduces conflicts later on.
    Author: Omar Magdy and Ratnajit Mukherjee
    """
    # For onnx export, conversion of Boxes to common format will be done in c++ side.
    if not onxx_export:
        bboxes[:, :, 0::2] *= image_size / width
        bboxes[:, :, 1::2] *= image_size / height

    return bboxes, clses, scores


def centernet_post_process(bboxes, clses, scores, image_size):
    detections = []

    for i in range(bboxes.shape[0]):
        container = Container(boxes=bboxes[i], labels=clses[i], scores=scores[i])
        container.img_width = image_size
        container.img_height = image_size
        detections.append(container)

    return detections
