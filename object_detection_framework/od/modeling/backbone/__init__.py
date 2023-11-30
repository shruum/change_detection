from od.modeling import registry
from .hardnet import HarDNet

__all__ = ["build_backbone", "HarDNet"]


def build_backbone(cfg):
    return registry.BACKBONES[cfg.MODEL.BACKBONE.NAME](
        cfg, cfg.MODEL.BACKBONE.PRETRAINED
    )
