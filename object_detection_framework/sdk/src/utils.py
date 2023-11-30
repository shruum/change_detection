import glob
import os
import sys
import yaml

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../.."))
# pylint: disable=wrong-import-position
from od.data.datasets.dataset_class_names import dataset_classes
from od.default_config import cfg

# pylint: enable=wrong-import-position

SUB_CFG_DICT = {}

try:
    from od.default_config import centernet_cfg

    SUB_CFG_DICT["CenterNetHead"] = centernet_cfg
except ImportError:
    pass

try:
    from od.default_config import ssd_cfg

    SUB_CFG_DICT["SSDBoxHead"] = ssd_cfg
except ImportError:
    pass

try:
    from od.default_config import thundernet_cfg

    SUB_CFG_DICT["ThunderNetHead"] = thundernet_cfg
except ImportError:
    pass

try:
    from od.default_config import yolo_cfg

    SUB_CFG_DICT["Yolov2Head"] = yolo_cfg
    SUB_CFG_DICT["Yolov3Head"] = yolo_cfg
except ImportError:
    pass


def config_load(cfg_file, options=None):
    if options is None:
        options = []
    with open(cfg_file) as file:
        head = yaml.load(file, Loader=yaml.FullLoader)["MODEL"]["HEAD"]["NAME"]
    sub_cfg = SUB_CFG_DICT[head]
    cfg.merge_from_other_cfg(sub_cfg)
    cfg.merge_from_file(cfg_file)
    cfg.merge_from_list(options)
    cfg.freeze()

    return cfg


def get_class_names(dataset_type):
    try:
        class_names = dataset_classes[dataset_type]
    except KeyError:
        raise NotImplementedError(dataset_type + " dataset not supported.")
    return class_names


def find_files_with_extensions(where, extensions, recursive=False):
    """Return all the files from `where` which match one of the `extensions`."""
    image_paths = []
    for ext in extensions:
        image_paths += glob.glob(os.path.join(where, "*" + ext), recursive=recursive)
    return image_paths
