from .model_zoo import get_model
from .model_store import get_model_file
from .base import *
from .fcn import *
from .psp import *
from .encnet import *
from .dfanet import *
from .fast_laddernet_se import *
from .fpn import *
from .mobilenet_v3_seg_head import *
from .multi_head_net import *
from .danet import *
from .hrnet import get_hr_net
from .bisenet import get_bisenet
from .bisenet_spatial_bottleneck import get_bisenet_sp_b
from .bisenet_cc1 import get_bisenet_cc1
from .edanet import get_edanet
from .fastscnn import get_fast_scnn
from .enet import get_enet
from .contextnet import get_contextnet
from .fchardnet import get_fchardnet
from .icnet import get_icnet
from .cabinet_variants import (
    get_rgpnet,
    get_cabinet_v5,
    get_cabinet_v6,
    get_cabinet_slim,
    get_cabinet_v4_big,
)
from .swiftnet import get_swiftnet


def get_segmentation_model(name, **kwargs):
    from .fcn import get_fcn

    models = {
        "icnet": get_icnet,
        "bisenet": get_bisenet,
        "bisenet_spatial_bottleneck": get_bisenet_sp_b,
        "bisenet_cc1": get_bisenet_cc1,
        "contextnet": get_contextnet,
        "fcn": get_fcn,
        "pspnet": get_psp,
        "encnet": get_encnet,
        "dfanet": get_dfanet,
        "shelfnet": get_laddernet,
        "mobile_net_v3_head": get_mobile_net_v3_head,
        "multinet": get_multinet,
        "danet": get_danet,
        "hrnet": get_hr_net,
        "edanet": get_edanet,
        "fastscnn": get_fast_scnn,
        "enet": get_enet,
        "fchardnet": get_fchardnet,
        "cabinet_v4": get_rgpnet,
        "cabinet_v5": get_cabinet_v5,
        "cabinet_v6": get_cabinet_v6,
        "cabinet_slim": get_cabinet_slim,
        "cabinet_big": get_cabinet_v4_big,
        "rgpnet": get_rgpnet,
        "swiftnet": get_swiftnet,
    }
    return models[name.lower()](**kwargs)
