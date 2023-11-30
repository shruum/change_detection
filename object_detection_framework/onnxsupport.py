import torch.nn as nn
from torch import autograd
from enum import Enum
import torch


class NMSWithOnnxSupport(nn.Module):
    def __init__(
        self,
        cpu_nms,
        confidence_threshold,
        nms_threshold,
        max_per_class,
        max_per_image,
        is_normalized,
    ):
        super(NMSWithOnnxSupport, self).__init__()

        class OnnxNMS(autograd.Function):
            confidence_threshold = None
            nms_threshold = None
            max_per_class = None
            max_per_image = None
            cpu_nms = None

            @staticmethod
            def forward(ctx, scores, boxes):
                return scores

            @staticmethod
            def symbolic(g, scores, boxes):
                return g.op(
                    "nms_TRT",
                    scores,
                    boxes,
                    confidence_threshold_f=OnnxNMS.confidence_threshold,
                    nms_threshold_f=OnnxNMS.nms_threshold,
                    max_per_image_i=OnnxNMS.max_per_image,
                    max_per_class_i=OnnxNMS.max_per_class,
                    is_normalized_i=OnnxNMS.is_normalized,
                )

        OnnxNMS.cpu_nms = cpu_nms
        OnnxNMS.confidence_threshold = confidence_threshold
        OnnxNMS.max_per_class = max_per_class
        OnnxNMS.max_per_image = max_per_image
        OnnxNMS.nms_threshold = nms_threshold
        OnnxNMS.is_normalized = is_normalized

        self.implementation = OnnxNMS()

    def forward(self, scores_boxes):
        scores = scores_boxes[0]
        boxes = scores_boxes[1]
        boxes = boxes.reshape([1, -1, 1, 4])
        res = self.implementation.apply(scores, boxes)
        return res


class ThundernetHelper(nn.Module):
    def __init__(self):
        super(ThundernetHelper, self).__init__()

        class OnnxThundernetHelper(autograd.Function):
            @staticmethod
            def forward(ctx, scores, boxes, operation, stride, img_width, img_height):
                if operation == ThundernetHelperOperations.ANCHOR.value:
                    second_index = (
                        int(scores.size(0))
                        * int(scores.size(1))
                        * int(scores.size(2))
                        * int(scores.size(3))
                    )
                    boxes = boxes.new(1, second_index, 4).zero_()

                return boxes

            @staticmethod
            def symbolic(
                g, scores, boxes, operation, stride=1, img_width=511, img_height=511
            ):
                return g.op(
                    "thundernet_helper_TRT",
                    scores,
                    boxes,
                    operation_i=operation,
                    stride_i=stride,
                    width_i=img_width,
                    height_i=img_height,
                )

        self.implementation = OnnxThundernetHelper()

    def forward(
        self, scores, boxes, operation, stride=1, img_width=511, img_height=511
    ):
        res = self.implementation.apply(
            scores, boxes, operation, stride, img_width, img_height
        )
        return res


# This enum is matched with thundernet_helper in tensorrt, Do NOT edit order.
class ThundernetHelperOperations(Enum):
    ANCHOR = 0
    ADVANCED_SLICING = 1
    PICK_BOXES = 2
    BBOX_INV_TRANSFORM = 3
    CLIP_BOXES = 4


class CenternetHelper(nn.Module):
    def __init__(self):
        super(CenternetHelper, self).__init__()

        class OnnxCenternetHelper(autograd.Function):
            @staticmethod
            def forward(ctx, in1, in2, operation, attr1, attr2, attr3):
                if operation == CenternetHelperOperations.GATHER_FEAT.value:
                    select = in2.size(1)
                    return in1[:, 0:select, :]
                if operation == CenternetHelperOperations.TOPK.value:
                    return in1[:, 0:5, 0:100].squeeze().mean(2)
                if operation == CenternetHelperOperations.PREPARE_OUTPUT.value:
                    return torch.cat([in1, in2], 2)

                return in1

            @staticmethod
            def symbolic(g, in1, in2, operation, attr1=1, attr2=1, attr3=1):
                return g.op(
                    "centernet_helper_TRT",
                    in1,
                    in2,
                    operation_i=operation,
                    attr1_i=attr1,
                    attr2_i=attr2,
                    attr3_i=attr3,
                )

        self.implementation = OnnxCenternetHelper()

    def forward(self, in1, in2, operation, attr1=1, attr2=1, attr3=1):
        res = self.implementation.apply(in1, in2, operation, attr1, attr2, attr3)
        return res


# This enum is matched with centernet_helper in tensorrt, Do NOT edit order.
class CenternetHelperOperations(Enum):
    CENTERNET_NMS = 0
    GATHER_FEAT = 1
    TOPK = 2
    PREPARE_OUTPUT = 3


class SORT(nn.Module):
    def __init__(self):
        super(SORT, self).__init__()

        class OnnxSORT(autograd.Function):
            @staticmethod
            def forward(ctx, scores):
                return scores

            @staticmethod
            def symbolic(g, scores):
                return g.op("sort_TRT", scores)

        self.implementation = OnnxSORT()

    def forward(self, scores):
        res = self.implementation.apply(scores)
        return res


class SOFT_NMS_TRT(nn.Module):
    def __init__(self):
        super(SOFT_NMS_TRT, self).__init__()

        class OnnxSOFT_NMS_TRT(autograd.Function):
            @staticmethod
            def forward(ctx, scores, boxes, threshold, sigma, nt, method):
                return scores

            @staticmethod
            def symbolic(g, scores, boxes, threshold, sigma, nt, method):
                return g.op(
                    "soft_nms_TRT",
                    scores,
                    boxes,
                    threshold_f=threshold,
                    sigma_f=sigma,
                    nt_f=nt,
                    method_i=method,
                )

        self.implementation = OnnxSOFT_NMS_TRT()

    def forward(self, scores, boxes, threshold, sigma, nt, method):
        res = self.implementation.apply(scores, boxes, threshold, sigma, nt, method)
        return res


class NOP(nn.Module):
    def __init__(self):
        super(NOP, self).__init__()

        class OnnxNOP(autograd.Function):
            @staticmethod
            def forward(ctx, scores):
                return scores

            @staticmethod
            def symbolic(g, scores):
                return g.op("nop_TRT", scores)

        self.implementation = OnnxNOP()

    def forward(self, scores):
        res = self.implementation.apply(scores)
        return res
