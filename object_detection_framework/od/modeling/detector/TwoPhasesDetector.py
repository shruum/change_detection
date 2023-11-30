from torch import nn

from od.modeling.backbone import build_backbone
from od.modeling.head import build_head
from od.modeling.detector.build_customized_func import build_customized_func

from od.modeling.detector.choose_features import choose_features


class TwoPhasesDetector(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.backbone = build_backbone(cfg)
        self.head = build_head(cfg)
        self.features_for_head = choose_features(cfg)
        self.customized_func = build_customized_func(cfg)

    def forward(self, images, targets=None):
        features = self.backbone(images)
        features = self.features_for_head(features)
        kwargs = self.customized_func(images)
        detections, detector_losses = self.head(features, targets, **kwargs)
        if self.training:
            return detector_losses
        return detections
