import logging
import numpy as np
import os
import sys

from utils import imread, imwrite

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_PATH, "cabinet_star"))

from cabinet_star.sdk.src.engine import Engine


class Segmenter:
    def __init__(self, cache_dir):
        logging.info("Initializing the segmenter...")
        self.engine = Engine(
            ckpt="/data/nie/teams/arl/models_checkpoints/crd/2.0/segmenter/model_best.pth.tar",
            backbone="resnet50",
            model="bisenet",
            output_write=False,
            verbose=False,
        )
        self.engine.load_model()
        self.cache_dir = os.path.join(cache_dir, "segmentation")
        os.makedirs(self.cache_dir, exist_ok=True)
        logging.info("segmenter loaded.")

    def segment(self, img_path):
        basename, _ = os.path.splitext(os.path.basename(img_path))
        cache_filename = os.path.join(self.cache_dir, basename + ".png")
        if os.path.exists(cache_filename):
            seg_map = imread(cache_filename)
        else:
            seg_map = np.array(self.engine.infer_img(img_path))
            imwrite(cache_filename, seg_map)
        return seg_map
