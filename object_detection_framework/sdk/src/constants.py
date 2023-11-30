import os

HOME_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")

DEFAULT_CFG = os.path.join(HOME_DIR, "model/config.yaml")
DEFAULT_CFG_OPTIONS = ()
DEFAULT_CKPT = os.path.join(HOME_DIR, "model/checkpoints.pth")
DEFAULT_DATASET_TYPE = "voc"
DEFAULT_INPUT_DIR = os.path.join(HOME_DIR, "input")
DEFAULT_OUTPUT_DIR = os.path.join(HOME_DIR, "output")
DEFAULT_OUTPUT_FORMAT = "img"
DEFAULT_OUTPUT_WRITE = True
DEFAULT_SCORE_THRESHOLD = 0.25
DEFAULT_VERBOSE = False
SUPPORTED_IMAGE_FORMATS = [".png", ".jpg", ".JPG"]
