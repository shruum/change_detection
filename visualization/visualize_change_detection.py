# Copyright (C) 2020 by NavInfo Europe B.V. The Netherlands - All rights reserved
# Information classification: Confidential
# This content is protected by international copyright laws.
# Reproduction and distribution is prohibited without written permission.

import csv
import cv2
import glob
import json
import logging
import numpy as np
import os
import sys

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_PATH))
import bbox_visualizer.bbox_visualizer as bbv

sys.path.insert(0, os.path.join(SCRIPT_PATH, ".."))
from utils import get_change_per_frame

sys.path.insert(0, os.path.join(SCRIPT_PATH, "..", "nie_utilities"))
from data_processing_utils import (
    get_detections_from_json_dict,
    load_json,
    get_basename_no_ext,
    xyxy_to_xywh,
    xywh_to_xyxy,
    find_match_else_add,
)


class VisualizeChangeDetection:
    def __init__(
        self, run1_paths, run2_paths, cache_dir,
    ):
        self.run1_paths = np.array(run1_paths)
        self.run2_paths = np.array(run2_paths)
        self.cache_dir = cache_dir
        self.colors = [
            [105, 185, 84],
            [196, 92, 162],
            [89, 131, 70],
            [123, 116, 204],
            [181, 161, 69],
            [76, 181, 188],
            [206, 84, 88],
            [196, 113, 56],
        ]
        self.class_list = ["Traffic Sign/Light", "Arrow"]

        logging.info("Visualization and Comparison of two runs")

    def _get_changes(self):
        run1_length = len(self.run1_paths)
        run2_length = len(self.run2_paths)
        max_frames = int(np.max([run1_length, run2_length]))
        changes_list = [[] for _ in range(max_frames)]
        for index in range(len(self.change_array)):
            frame_id = self.change_array[index, 0]
            change_frame_list = self.change_array[index, 1]
            class_frame_list = self.change_array[index, 2]
            changes_list[frame_id].append(change_frame_list)
            changes_list[frame_id].append(class_frame_list)
        return changes_list

    def _get_detections_from_nie_json(self, class_type="arrow_detections", run="run1"):
        results_file_list = glob.glob(
            os.path.join(self.cache_dir, class_type, run, "*.json")
        )
        detection_dict = {}
        for results_file in results_file_list:
            result_dict = load_json(results_file)
            detection_list = get_detections_from_json_dict(result_dict)
            basename = get_basename_no_ext(results_file)
            detection_dict[basename] = detection_list
        return detection_dict

    def _get_detections_from_tracking(self, mapping_list, run_paths, run="run1"):
        detection_dict = {}
        run_path = os.path.join(self.cache_dir, "tracking", run)
        results_file_list = glob.glob(os.path.join(run_path, "results_track_*.json"))
        for results_file in results_file_list:
            with open(results_file) as file:
                track_file_dict = json.load(file)
            if track_file_dict["keep"]:
                track_object_dict = track_file_dict["objects"]
                class_id = track_file_dict["class_id"]
                class_id = ",".join(class_id)
                score = track_file_dict["score"]
                track_id = str(track_file_dict["instance_id"])
                for track_object in track_object_dict:
                    frame_id = track_object["frame_id"]
                    frame_start = mapping_list[frame_id][0]
                    frame_end = mapping_list[frame_id][1]
                    x0 = track_object["x0"]
                    y0 = track_object["y0"]
                    x1 = track_object["x1"]
                    y1 = track_object["y1"]
                    x0, y0, w, h = xyxy_to_xywh([x0, y0, x1, y1])
                    detection = [y0, x0, h, w, class_id, None, score, track_id]
                    for frame_id in range(frame_start, frame_end + 1):
                        basename = get_basename_no_ext(run_paths[frame_id])
                        if basename not in detection_dict:
                            detection_dict[basename] = []
                        detection_dict[basename].append(detection)
        return detection_dict

    def visualize_runs(self, info_window_width=1000):

        output_dir = os.path.join(self.cache_dir, "visualization")
        os.makedirs(output_dir, exist_ok=True)

        mapping_list_run1_path = os.path.join(self.cache_dir, "mapping_list_run1.npy")
        mapping_list_run2_path = os.path.join(self.cache_dir, "mapping_list_run2.npy")

        if os.path.isfile(mapping_list_run1_path) and os.path.isfile(
            mapping_list_run2_path
        ):

            # Get changes per frames
            changes_list = []
            final_changes = os.path.join(self.cache_dir, "final_changes.csv")
            with open(final_changes, "r") as file:
                reader = csv.reader(file)
                next(reader, None)
                for row in reader:
                    change_type = str(row[1])
                    major_category = str(row[2])
                    minor_category = str(row[3])
                    frame_start = int(row[-2])
                    frame_end = int(row[-1])
                    changes_list.append(
                        [
                            change_type,
                            major_category,
                            minor_category,
                            frame_start,
                            frame_end,
                        ]
                    )
            change_per_frame = get_change_per_frame(
                changes_list, 0, len(self.run1_paths)
            )

            # Get the detection data

            ## Traffic signs

            _, mapping_list_run1 = np.load(mapping_list_run1_path, allow_pickle=True)
            _, mapping_list_run2 = np.load(mapping_list_run2_path, allow_pickle=True)
            traffic_sign_dict_run1 = self._get_detections_from_tracking(
                mapping_list_run1, self.run1_paths, run="run1"
            )
            traffic_sign_dict_run2 = self._get_detections_from_tracking(
                mapping_list_run2, self.run2_paths, run="run2"
            )

            ## Arrows on the road

            arrows_dict_run1 = self._get_detections_from_nie_json(
                class_type="arrow_detections", run="run1"
            )
            arrows_dict_run2 = self._get_detections_from_nie_json(
                class_type="arrow_detections", run="run2"
            )

            # Draw everything

            for frame_index in range(len(self.run1_paths)):

                output_path = os.path.join(
                    os.path.join(output_dir, str(frame_index).zfill(6) + ".png")
                )

                if not os.path.isfile(output_path):

                    filepath_1 = self.run1_paths[frame_index]
                    filepath_2 = self.run2_paths[frame_index]

                    basename_1 = get_basename_no_ext(filepath_1)
                    basename_2 = get_basename_no_ext(filepath_2)

                    img_run_1 = cv2.imread(filepath_1)
                    img_run_2 = cv2.imread(filepath_2)

                    info_window = np.zeros((img_run_1.shape[0], info_window_width, 3))

                    # Obtain change status
                    (
                        _,
                        change_type_list,
                        change_major_label_list,
                        change_minor_label_list,
                    ) = change_per_frame[frame_index]
                    if change_type_list is not None:
                        info_window = self._write_change_state(
                            info_window,
                            change_type_list,
                            change_major_label_list,
                            change_minor_label_list,
                        )

                    # Display the detections

                    ## Traffic signs

                    img_run_1 = self._draw_bboxes(
                        img_run_1,
                        detection_list=traffic_sign_dict_run1.get(basename_1),
                        color_index=0,
                    )
                    img_run_2 = self._draw_bboxes(
                        img_run_2,
                        detection_list=traffic_sign_dict_run2.get(basename_2),
                        color_index=0,
                    )

                    ## Arrows on the road

                    img_run_1 = self._draw_bboxes(
                        img_run_1,
                        detection_list=arrows_dict_run1.get(basename_1),
                        color_index=1,
                    )
                    img_run_2 = self._draw_bboxes(
                        img_run_2,
                        detection_list=arrows_dict_run2.get(basename_2),
                        color_index=1,
                    )

                    # Stack the two runs horizontally and save final output

                    stacked_img = np.hstack([info_window, img_run_1, img_run_2])
                    stacked_img_resized = cv2.resize(stacked_img, (1920, 540))

                    cv2.imwrite(output_path, stacked_img_resized)

    def _write_change_state(
        self, image, change_type_list, change_major_class_list, change_minor_class_list
    ):
        for index, (change_type, major_class, minor_class) in enumerate(
            zip(change_type_list, change_major_class_list, change_minor_class_list)
        ):
            color_index, self.class_list = find_match_else_add(
                major_class, self.class_list
            )
            change_text = change_type.capitalize() + ":" + major_class
            if minor_class:
                change_text += " - " + minor_class
            font_color = self.colors[color_index]
            text_location = (10, 200 + index * 50)
            cv2.putText(
                image,
                change_text,
                text_location,
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                font_color,
                4,
                cv2.LINE_AA,
            )
        return image

    def _process_change_text(self, change_text):
        """
        Processes the different texts available in change state for single/multiple changes
        :param change_state: input list
        :return: processed dictionary
        """
        processed_change = {"sign_added": [], "sign_removed": []}
        if change_text[0] is not None:
            for status, label in zip(change_text[0], change_text[1]):
                if status == "Sign Added":
                    processed_change["sign_added"].append(label)
                elif status == "Sign Removed":
                    processed_change["sign_removed"].append(label)
        return processed_change

    def _draw_bboxes(self, image, detection_list, color_index=0, text_color=(0, 0, 0)):
        if detection_list is not None:
            # make a deep copy of an image
            image = image.copy()
            for detection in detection_list:
                y0, x0, h, w = detection[:4]
                track_id = detection[-1]
                class_name = str(detection[4])
                bbox_color = self.colors[color_index]

                bbox = xywh_to_xyxy([x0, y0, w, h])
                image = bbv.draw_rectangle(image, bbox, bbox_color=bbox_color)

                if track_id is not None:
                    cat_name = track_id + ";"
                else:
                    cat_name = ""
                cat_name += class_name

                image = bbv.add_label(
                    image,
                    cat_name,
                    bbox,
                    top=True,
                    text_bg_color=bbox_color,
                    text_color=text_color,
                )
        return image
