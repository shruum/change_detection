# -*- coding: utf-8 -*-
import base64
from io import BytesIO
import json
from PIL import Image
import numpy as np
import os
import sys

SCRIPTPATH = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(SCRIPTPATH))
import constants

sys.path.append(os.path.join(SCRIPTPATH, ".."))
from classifier import Classifier
from infer_single_image import infer


class Engine:
    def __init__(self):
        self.model_name = constants.DEFAULT_MODEL
        self.model_path = constants.DEFAULT_MODEL_PATH
        self.datatype = constants.DEFAULT_DATATYPE
        self.datadir = constants.DEFAULT_DATADIR
        self.top_num = constants.DEFAULT_TOPK
        self._model = None



    def load_model(self):
        self._model = Classifier(
            self.model_name,
            self.model_path,
            dataset_type=self.datatype,
            data_dir=self.datadir,
        )


    def inference(self, data):
        image_data = base64.b64decode(data)
        image = Image.open(BytesIO(image_data)).convert("RGB")
        class_list, score_list = infer(self._model, image, img_size=44, topk=self.top_num)
        if not isinstance(class_list, list):
            class_list = [class_list]
            score_list = [score_list]
        return self.convert_to_json(class_list, score_list)


    def release(self):
        pass


    def convert_to_json(self, class_list, score_list):
        result_dict = {}
        result_dict["cv_task"] = 0
        result_dict["obj_num"] = int(len(class_list))
        result_dict["top_num"] = int(self.top_num)
        object_list = []
        for object_index in range(self.top_num):
            class_id = class_list[object_index]
            score = score_list[object_index]
            class_index = np.nonzero(np.array(self._model.classes) == class_id)
            if len(class_index) > 0:
                object_dict = {}
                object_dict["f_code"] = int(class_index[0])
                object_dict["f_name"] = str(class_id)
                object_dict["f_conf"] = str(score)
                object_list.append(object_dict)
        result_dict["objects"] = object_list

        return json.dumps(result_dict, indent=4)