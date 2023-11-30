# -*- coding: utf-8 -*-

import logging
import os

from object_detection_framework.sdk.src.engine import Engine


class Detector:
    def __init__(self):
        logging.info("Initializing the detector...")
        detection_path = "/data/nie/teams/arl/models_checkpoints/crd/2.0/traffic_signs/"
        self.engine = Engine(
            config_file=os.path.join(detection_path, "config_file_new.yaml"),
            ckpt=os.path.join(detection_path, "detector_new.pth"),
            score_threshold=0.3,
            output_format="json_nie",
            dataset_type="ark",
        )
        self.engine.load_model()

    def detect(self, img_path, output_dir):
        basename, _ = os.path.splitext(os.path.basename(img_path))
        cache_filename = os.path.join(output_dir, basename + ".json")
        if not os.path.exists(cache_filename):
            self.engine.infer_img(img_path, output_dir)
