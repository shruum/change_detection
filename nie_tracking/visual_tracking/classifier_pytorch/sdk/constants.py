import os
SCRIPTPATH = os.path.dirname(os.path.realpath(__file__))
HOME_DIR = os.path.join(SCRIPTPATH, '..')
DEFAULT_MODEL = "resnet34"
DEFAULT_MODEL_PATH = os.path.join(HOME_DIR, "model", "checkpoint.pth")
DEFAULT_DATATYPE = ""
DEFAULT_DATADIR = os.path.join(HOME_DIR, "model")
DEFAULT_TOPK = 3