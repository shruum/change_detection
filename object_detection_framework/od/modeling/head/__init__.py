from od.modeling import registry
from od.modeling.head.centernet.centernet_head import CenterNetHead

__all__ = ["build_head", "CenterNetHead"]


def build_head(cfg):
    return registry.HEADS[cfg.MODEL.HEAD.NAME](cfg)
