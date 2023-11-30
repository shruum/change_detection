#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import json
import math
import numpy as np
import os
import random
from PIL import Image


class Track:
    def __init__(self, bbox, frame_id, track_id):

        self.track_id = track_id
        self.init_bbox = bbox
        self.frame_start = -1
        self.frame_end = -1
        self.last_frame_id = frame_id
        self.init_frame_id = frame_id
        self.is_end_track = False

        # Tracking fields
        self.detection_index_list = []
        self.tracking_bbox_list = []
        self.tracking_flag_list = []
        self.frame_id_list = []

        # Final output
        self.bbox_list = []
        self.is_detected_list = []
        self.detection_class_list = []
        self.detection_score_list = []
        self.classification_list = []
        self.classification_score_list = []

        self.result = []

    def __sort_list(self, x, sort_indices, unique_indices):
        x_sorted = x[sort_indices]
        x_unique = x_sorted[unique_indices]
        return x_unique

    def __concatenate_list(self, x, y):
        x = np.array(x)
        y = np.flipud(np.array(y))[:-1]
        ndim_x = x.ndim
        ndim_y = y.ndim
        if ndim_x == ndim_y:
            x_concat = np.concatenate((x, y))
        elif ndim_x < ndim_y:
            x_concat = y
        elif ndim_x > ndim_y:
            x_concat = x
        return x_concat

    def __concatenate_and_sort_list(self, x, y, sort_indices, unique_indices):
        x_concat = self.__concatenate_list(x, y)
        x_sorted = self.__sort_list(x_concat, sort_indices, unique_indices)
        return x_sorted

    def _concatenate_tracks(self, track):
        self.frame_id_list = self.__concatenate_list(
            self.frame_id_list, track.frame_id_list
        )
        sort_indices = np.argsort(self.frame_id_list)
        self.frame_id_list = self.frame_id_list[sort_indices]
        self.frame_id_list, unique_indices = np.unique(self.frame_id_list, True)
        self.detection_index_list = self.__concatenate_and_sort_list(
            self.detection_index_list,
            track.detection_index_list,
            sort_indices,
            unique_indices,
        )
        self.tracking_bbox_list = self.__concatenate_and_sort_list(
            self.tracking_bbox_list,
            track.tracking_bbox_list,
            sort_indices,
            unique_indices,
        )
        self.tracking_flag_list = self.__concatenate_and_sort_list(
            self.tracking_flag_list,
            track.tracking_flag_list,
            sort_indices,
            unique_indices,
        )
        self.bbox_list = self.__concatenate_and_sort_list(
            self.bbox_list, track.bbox_list, sort_indices, unique_indices
        )
        self.is_detected_list = self.__concatenate_and_sort_list(
            self.is_detected_list, track.is_detected_list, sort_indices, unique_indices
        )
        # Ensure correct type
        self.frame_id_list = self.frame_id_list.astype(int)
        self.detection_index_list = self.detection_index_list.astype(int)

    def _slice_track(self):
        frame_id_array = np.array(self.frame_id_list)
        end_index = np.nonzero(frame_id_array == self.last_frame_id)[0]
        end_index = end_index[0] if len(end_index) > 0 else 0
        self.frame_id_list = self.frame_id_list[:end_index]
        self.detection_index_list = self.detection_index_list[:end_index]
        self.tracking_bbox_list = self.tracking_bbox_list[:end_index]
        self.tracking_flag_list = self.tracking_flag_list[:end_index]
        self.bbox_list = self.bbox_list[:end_index]

    def _save_json_output(self, output_dir, track_id, img_width, img_height):
        output_path = os.path.join(
            output_dir, "results_track_" + str(track_id).zfill(4) + ".json"
        )
        track_file_dict = dict()
        track_file_dict["instance_id"] = int(track_id)
        track_file_dict["width"] = int(img_width)
        track_file_dict["height"] = int(img_height)
        track_file_dict["objects"] = []
        for object_index in range(len(self.bbox_list)):
            object_dict = dict()
            object_dict["frame_id"] = int(self.frame_id_list[object_index])
            bbox = self.bbox_list[object_index]
            object_dict["x0"] = int(bbox[1])
            object_dict["y0"] = int(bbox[0])
            object_dict["x1"] = int(bbox[3])
            object_dict["y1"] = int(bbox[2])
            object_dict["detection_class"] = str(
                self.detection_class_list[object_index]
            )
            object_dict["detection_score"] = float(
                self.detection_score_list[object_index]
            )
            object_dict["is_detected"] = bool(self.is_detected_list[object_index])
            if len(self.classification_list) > 0:
                object_dict["classification_list"] = self.classification_list[
                    object_index
                ]
                object_dict[
                    "classification_score_list"
                ] = self.classification_score_list[object_index]
            track_file_dict["objects"].append(object_dict)
        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(track_file_dict, file, ensure_ascii=False, indent=3)

    def _check_video(self, input_data):
        if not isinstance(input_data, list):
            cap = cv2.VideoCapture(input_data)
        else:
            cap = []
        return cap

    def _get_crop(self, input_data, frame, frame_index, cap):
        if cap:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ret, image = cap.read()
        else:
            image = cv2.imread(input_data[frame])
        bbox = self.bbox_list[frame_index]
        y0 = int(math.floor(bbox[0]))
        x0 = int(math.floor(bbox[1]))
        y1 = int(math.ceil(bbox[2]))
        x1 = int(math.ceil(bbox[3]))
        if y0 < 0:
            y0 = 0
        if x0 < 0:
            x0 = 0
        if y1 >= image.shape[0]:
            y1 = image.shape[0] - 1
        if x1 >= image.shape[1]:
            x1 = image.shape[1] - 1
        im = image[y0:y1, x0:x1, :]
        return im

    def _save_crops(self, crop_path, input_data):
        cap = self._check_video(input_data)
        for frame_index, frame in enumerate(self.frame_id_list):
            im = self._get_crop(input_data, frame, frame_index, cap)
            crop_name = f"{self.track_id}_{frame:05d}.png"
            crop_name = os.path.join(crop_path, crop_name)
            cv2.imwrite(crop_name, im)

    def _get_probability(self, frame_id, classification_window):
        frame_difference = np.abs(self.init_frame_id - frame_id)
        if classification_window > 0:
            probability = 1 / (2 ** (frame_difference // classification_window))
        else:
            probability = 1.0
        return probability

    def _classify(self, input_data, classifier, classification_window):
        cap = self._check_video(input_data)
        class_list = []
        score_list = []
        for frame_index, frame_id in enumerate(self.frame_id_list):
            probability = self._get_probability(frame_id, classification_window)
            if probability >= random.random():
                im = self._get_crop(input_data, frame_id, frame_index, cap)
                im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                class_list_crop, score_list_crop = classifier.infer_image(
                    Image.fromarray(im), 64, topk=3
                )
            else:
                class_list_crop = [-1, -1, -1]
                score_list_crop = [-1, -1, -1]
            class_list.append(class_list_crop)
            score_list.append(score_list_crop)
        return class_list, score_list
