# ------------------------------------------------------------------------------
# Portions of this code are from
# CornerNet (https://github.com/princeton-vl/CornerNet)
# Copyright (c) 2019, NavInfo Europe
# NOTE:Implementation of CenterNet Head in Detection as a service Framework
# ------------------------------------------------------------------------------

import os
import torch
import torch.nn as nn
from od.modeling import registry
from od.modeling.head.centernet.centernet_predictor import CenterNetHeadPredictor
from od.modeling.head.centernet.CenterNetLossCalculator import CtdetLoss
from od.modeling.head.centernet.centernet_inference import centernet_eval_process
from od.modeling.head.centernet.decode import centernet_post_process
from onnxsupport import CenternetHelperOperations, CenternetHelper


@registry.HEADS.register("CenterNetHead")
class CenterNetHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # initializing the CenterNet Head
        self.centernet_pred = CenterNetHeadPredictor(cfg)
        self.centernet_loss = CtdetLoss(cfg)

    def forward(self, features, targets=None):
        # now we pass the
        input_feature = features[0]
        if input_feature is None:
            assert "ERROR: Input Feature not found in backbone"
        output_features = self.centernet_pred(input_feature)
        if self.training:
            return self._forward_train(output_features, targets)
        else:
            return self._forward_test(output_features[0])

    def _forward_train(self, output_features, targets):
        loss_stats = self.centernet_loss(output_features, targets)
        return output_features[0], loss_stats

    def _forward_test(self, output_features):
        onnx_export = self.cfg.EXPORT == "onnx"
        bboxes, clses, scores, width, height = centernet_eval_process(
            output_features, self.cfg.INPUT.IMAGE_SIZE, onnx_export
        )

        if onnx_export:
            # Tensorrt expect one tensor with structure of [minx, miny, maxx, maxy, score, class]
            # So final result dimensions should be [batch_size, num_detections, 6]
            # Before combine results, boxes needs to be converted to generic format. That requires partial
            # assignment. Onnx export cannot recognize this assignment so this conversion needs to be done in
            # c++ side
            scores_cls_combined = torch.cat(
                [scores.unsqueeze(2), clses.unsqueeze(2)], 2
            )
            detections = CenternetHelper()(
                bboxes,
                scores_cls_combined,
                CenternetHelperOperations.PREPARE_OUTPUT.value,
                self.cfg.INPUT.IMAGE_SIZE,
                width,
                height,
            )
        else:
            detections = centernet_post_process(
                bboxes, clses, scores, self.cfg.INPUT.IMAGE_SIZE
            )

        return detections, {}
