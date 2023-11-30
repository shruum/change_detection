#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
import glob
import os
import numpy as np
import sys

CURRENT_DIR = os.path.dirname(__file__)
sys.path.insert(
    0, os.path.join(CURRENT_DIR, "nie_tracking", "visual_tracking"),
)
from tracker import Tracker as Engine

sys.path.insert(
    0, os.path.join(CURRENT_DIR, "nie_utilities"),
)
from data_processing_utils import get_basename_no_ext, find_match

class Tracker:
    def __init__(self, classifier_path=None):
        logging.info("Initializing the tracker...")
        if classifier_path is None:
            classifier_path = "/data/nie/teams/arl/models_checkpoints/crd/2.0/traffic_signs/classifier.pth"
            classifier_datatype = "crd_181"
        else:
            classifier_datatype = "custom"
        classifier_datadir = os.path.dirname(classifier_path)
        self.engine = Engine(
            classifier_path=classifier_path,
            upper_conf_threshold=0.45,
            min_length=8,
            max_length=200,
            run_classifier=True,
            classification_window=10,
            classifier_datatype=classifier_datatype,
            classifier_datadir=classifier_datadir,
        )

    def _get_detection_and_image_list(self, json_path_list, image_path_list):
        # Get image and detection files
        json_path_list_ordered = []
        image_path_list_ordered = []
        json_basename_list = []
        for json_path in json_path_list:
            json_basename_list.append(get_basename_no_ext(json_path))
        json_basename_list = np.array(json_basename_list)
        for image_path in image_path_list:
            image_basename = get_basename_no_ext(os.path.normpath(image_path))
            file_id = find_match(image_basename, json_basename_list)
            if file_id is not None:
                json_path_list_ordered.append(json_path_list[file_id])
                image_path_list_ordered.append(image_path)
        return json_path_list_ordered, image_path_list_ordered

    def track(self, image_path_list, detection_dir, output_dir):
        json_path_list = sorted(glob.glob(os.path.join(detection_dir, "*.json")))
        json_path_list, image_path_list = self._get_detection_and_image_list(json_path_list, image_path_list)
        self.engine.tracking(image_path_list, json_path_list, output_dir)