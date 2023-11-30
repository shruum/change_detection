from yacs.config import CfgNode as CN

_C = CN(new_allowed=True)

# ---------------------------------------------------------------------------------#
# ONNX EXPORT
# ---------------------------------------------------------------------------------#
_C.EXPORT = "none"

# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN(new_allowed=True)
# Image size
_C.INPUT.IMAGE_SIZE = 300
# Values to be used for image normalization, RGB layout
_C.INPUT.PIXEL_MEAN = [123, 117, 104]
_C.INPUT.MAX_NUM_GT_BOXES = 20

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN(new_allowed=True)
_C.DATASETS.PATH = ""
# List of the dataset names for training, as present in paths_catalog.py
_C.DATASETS.TRAIN = ()
# List of the dataset names for testing, as present in paths_catalog.py
_C.DATASETS.TEST = ()

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATA_LOADER = CN(new_allowed=True)
# Number of data loading threads
_C.DATA_LOADER.NUM_WORKERS = 8
_C.DATA_LOADER.PIN_MEMORY = True
_C.DATA_LOADER.INCLUDE_BACKGROUND = True
_C.DATA_LOADER.MULTISCALE = False
# ---------------------------------------------------------------------------- #
# Scheduler
# ---------------------------------------------------------------------------- #
_C.SCHEDULER = CN(new_allowed=True)
_C.SCHEDULER.TYPE = "WarmupMultiStepLR"

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
_C.MODEL = CN(new_allowed=True)
_C.MODEL.META_ARCHITECTURE = "TwoPhasesDetector"
_C.MODEL.DEVICE = "cuda"
# match default boxes to any ground truth with jaccard overlap higher than a threshold (0.5)
_C.MODEL.THRESHOLD = 0.5
_C.MODEL.NUM_CLASSES = 21
# Hard negative mining
_C.MODEL.NEG_POS_RATIO = 3
_C.MODEL.CENTER_VARIANCE = 0.1
_C.MODEL.SIZE_VARIANCE = 0.2

# ---------------------------------------------------------------------------- #
# Backbone
# ---------------------------------------------------------------------------- #
_C.MODEL.BACKBONE = CN(new_allowed=True)
_C.MODEL.BACKBONE.NAME = "vgg"
_C.MODEL.BACKBONE.OUT_CHANNELS = (512, 1024, 512, 256, 256, 256)
_C.MODEL.BACKBONE.PRETRAINED = True
_C.MODEL.BACKBONE.WIDTH_MULT = 1.0
_C.MODEL.BACKBONE.FEATURE_MAPS = []
_C.MODEL.BACKBONE.EXTRA = True
_C.MODEL.BACKBONE.FEATURE_INDEX = []
_C.MODEL.BACKBONE.WEIGHTS_PATH = ""
# ---------------------------------------------------------------------------- #
# HEAD
# ---------------------------------------------------------------------------- #
_C.MODEL.HEAD = CN(new_allowed=True)
_C.MODEL.HEAD.NAME = ""  #'SSDBoxHead'

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN(new_allowed=True)
# train configs
_C.SOLVER.NAME = "SGD_optimizer"
_C.SOLVER.MAX_ITER = 120000
_C.SOLVER.LR_STEPS = [80000, 100000]
_C.SOLVER.GAMMA = 0.1
_C.SOLVER.BATCH_SIZE = 32
_C.SOLVER.LR = 1e-3
_C.SOLVER.MOMENTUM = 0.9
_C.SOLVER.WEIGHT_DECAY = 5e-4
_C.SOLVER.WARMUP_FACTOR = 1.0 / 3
_C.SOLVER.WARMUP_ITERS = 500


# ---------------------------------------------------------------------------------#
# LOGGER
# ---------------------------------------------------------------------------------#
_C.LOGGER = CN(new_allowed=True)
_C.LOGGER.NAME = "SSD.trainer"
_C.LOGGER.DEBUG_MODE = False
# ---------------------------------------------------------------------------- #
# Specific test options
# ---------------------------------------------------------------------------- #
_C.TEST = CN(new_allowed=True)
_C.TEST.NMS_THRESHOLD = 0.45
_C.TEST.CONFIDENCE_THRESHOLD = 0.01
_C.TEST.MAX_PER_CLASS = -1
_C.TEST.MAX_PER_IMAGE = 100
_C.TEST.BATCH_SIZE = 1
_C.TEST.GET_INF_TIME = True

_C.OUTPUT_DIR = "outputs\ssd"
