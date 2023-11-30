#! /usr/bin/env python
# -*- coding: utf-8 -*-

r"""Generate tracking results for videos"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import glob
import os
import sys
import numpy as np
from tqdm import tqdm

from tracker_utils import (
    convert_output_pytracking,
    calculate_iou,
    read_detection_file_json,
    predict_trajectory,
    get_detection_and_image_list,
)
from track import Track

CURRENT_DIR = os.path.dirname(__file__)
PARENT_DIR = os.path.join(CURRENT_DIR, "..", "..", "..", "..", "..")

DEFAULT_PARAMS = {
    "method": "DiMP",
    "input_type": "image",
    "output_filename": "results",
    "num_future_frames": 15,
    "min_length": 8,
    "max_length": 500,
    "lower_conf_threshold": 0.3,
    "upper_conf_threshold": 0.5,
    "iou_threshold": 0.3,
    "max_relative_diff": 0,
    "img_height": 1080,
    "img_width": 1920,
    "crop_window": "0,250,75,150",
    "min_bbox_area": 300,
    "min_displacement": 1e-3,
    "save_crop_images": False,
    "run_classifier": False,
    "use_detection": False,
    "classifier_datatype": None,
    "classifier_datadir": ".",
    "classifier_image_size": 64,
    "end_threshold": 3,
    "classification_window": 0,
}


class Tracker:
    def __init__(
        self,
        classifier_path=[],
        method=DEFAULT_PARAMS["method"],
        input_type=DEFAULT_PARAMS["input_type"],
        output_filename=DEFAULT_PARAMS["output_filename"],
        num_future_frames=DEFAULT_PARAMS["num_future_frames"],
        min_length=DEFAULT_PARAMS["min_length"],
        max_length=DEFAULT_PARAMS["max_length"],
        lower_conf_threshold=DEFAULT_PARAMS["lower_conf_threshold"],
        upper_conf_threshold=DEFAULT_PARAMS["upper_conf_threshold"],
        iou_threshold=DEFAULT_PARAMS["iou_threshold"],
        max_relative_diff=DEFAULT_PARAMS["max_relative_diff"],
        img_height=DEFAULT_PARAMS["img_height"],
        img_width=DEFAULT_PARAMS["img_width"],
        crop_window=DEFAULT_PARAMS["crop_window"],
        min_bbox_area=DEFAULT_PARAMS["min_bbox_area"],
        min_displacement=DEFAULT_PARAMS["min_displacement"],
        save_crop_images=DEFAULT_PARAMS["save_crop_images"],
        run_classifier=DEFAULT_PARAMS["run_classifier"],
        use_detection=DEFAULT_PARAMS["use_detection"],
        classifier_datatype=DEFAULT_PARAMS["classifier_datatype"],
        classifier_datadir=DEFAULT_PARAMS["classifier_datadir"],
        classifier_image_size=DEFAULT_PARAMS["classifier_image_size"],
        end_threshold=DEFAULT_PARAMS["end_threshold"],
        classification_window=DEFAULT_PARAMS["classification_window"],
    ):

        self.classifier_path = classifier_path
        self.method = method
        self.input_type = input_type
        self.output_filename = output_filename
        self.num_future_frames = num_future_frames
        self.min_length = min_length
        self.max_length = max_length
        if use_detection:
            self.lower_conf_threshold = lower_conf_threshold
        else:
            self.lower_conf_threshold = 0
        self.upper_conf_threshold = upper_conf_threshold
        self.iou_threshold = iou_threshold
        self.max_relative_diff = max_relative_diff
        self.img_height = img_height
        self.img_width = img_width
        self.crop_window = [int(item) for item in crop_window.split(",")]
        self.min_bbox_area = min_bbox_area
        self.min_displacement = min_displacement
        self.save_crop_images = save_crop_images
        self.run_classifier = run_classifier
        self.use_detection = use_detection
        self.classifier_datatype = classifier_datatype
        self.classifier_datadir = classifier_datadir
        self.classifier_image_size = classifier_image_size
        self.end_threshold = end_threshold
        self.classification_window = classification_window

        self.tracker, self.Sequence = self._init_pytracking()
        self.classifier = self._init_classifier()

    def _init_pytracking(self):
        PYTRACKING_PATH = os.path.join(CURRENT_DIR, "pytracking")
        sys.path.insert(0, PYTRACKING_PATH)
        sys.path.insert(0, os.path.join(PYTRACKING_PATH, "pytracking"))
        from evaluation import Sequence, Tracker

        if self.method == "ATOM":
            tracker_name = "atom"
            tracker_param = "default_no_rot"
        elif self.method == "DiMP":
            tracker_name = "dimp"
            tracker_param = "dimp50_no_rot"
        tracker = Tracker(tracker_name, tracker_param)
        return tracker, Sequence

    def _init_classifier(self):
        if self.classifier_path:
            CLASSIFIER_PATH = os.path.join(CURRENT_DIR, "classifier_pytorch")
            sys.path.insert(0, CLASSIFIER_PATH)
            from classifier import Classifier

            return Classifier(
                "resnet34",
                self.classifier_path,
                dataset_type=self.classifier_datatype,
                data_dir=self.classifier_datadir,
            )

    def _run_pytracking(self, image_track, bb):
        init_bb = [int(bb[1]), int(bb[0]), int(bb[3] - bb[1]), int(bb[2] - bb[0])]
        init_bb = np.array(init_bb)
        init_bb = init_bb[np.newaxis, :]
        output = self.tracker.run(
            self.Sequence("image_track", np.array(image_track), init_bb)
        )
        trajectory = convert_output_pytracking(output)
        return trajectory, output["flag"]

    def _get_image_track_from_image(self, track, image_list, direction="forwards"):
        image_track = []
        frame_id_list = []
        if direction == "forwards":
            frame_start = track.last_frame_id
            frame_end = track.last_frame_id + self.num_future_frames
        elif direction == "backwards":
            frame_start = track.last_frame_id - self.num_future_frames + 1
            frame_end = track.last_frame_id + 1
        for frame_id in range(frame_start, frame_end):
            if frame_id < 0 or frame_id >= len(image_list):
                track.is_end_track = True
                continue
            image = cv2.imread(image_list[frame_id])
            image_track.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            frame_id_list.append(frame_id)
        return image_track, frame_id_list, track

    def _get_image_track_from_video(self, track, video_file, direction="forwards"):
        image_track = []
        frame_id_list = []
        cap = cv2.VideoCapture(video_file)
        if direction == "forwards":
            frame_start = int(track.last_frame_id)
        elif direction == "backwards":
            frame_start = int(track.last_frame_id - self.num_future_frames + 1)
            if frame_start < 0:
                frame_start = 0
                track.is_end_track = True
        frame_id = frame_start
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)
        while len(image_track) != self.num_future_frames:
            ret, image = cap.read()
            if ret:
                image_track.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                frame_id_list.append(frame_id)
                frame_id += 1
            else:
                track.is_end_track = True
                break
        return image_track, frame_id_list, track

    def _get_tracking_prediction(self, track, direction="forwards"):
        if self.input_type == "video":
            image_track, frame_id_list, track = self._get_image_track_from_video(
                track, self.input_data, direction
            )
        elif self.input_type == "image":
            image_track, frame_id_list, track = self._get_image_track_from_image(
                track, self.input_data, direction
            )
        if direction == "backwards":
            image_track.reverse()
            frame_id_list.reverse()

        if len(image_track) > 0:
            trajectory, flag_list = self._run_pytracking(
                image_track, track.init_bbox  # [y1, x1, y4, x4]
            )
            if len(trajectory) != self.num_future_frames:
                track.is_end_track = True
            if len(track.bbox_list) > 0:
                frame_id_list = frame_id_list[1:]
                flag_list = flag_list[1:]
                trajectory = trajectory[1:]
            for frame_id in frame_id_list:
                track.frame_id_list.append(frame_id)
            for flag in flag_list:
                track.tracking_flag_list.append(flag)
            for bbox in trajectory:
                track.tracking_bbox_list.append(
                    [bbox.y, bbox.x, bbox.y + bbox.height, bbox.x + bbox.width]
                )
        else:
            track.is_end_track = True

        return track

    def _match_detections(self, track):
        for bbox_index in range(len(track.bbox_list), len(track.tracking_bbox_list)):
            tracking_bbox = track.tracking_bbox_list[bbox_index]
            frame_id = track.frame_id_list[bbox_index]
            detection_index_list = np.nonzero(
                frame_id == np.array(self.detection_frame_id_list)
            )[0]
            matched_detection_index = -1
            if len(detection_index_list) > 0:
                for detection_index in detection_index_list:
                    confidence = self.detection_confidence_list[detection_index]
                    detection_bbox = self.detection_bbox_list[detection_index]
                    if (
                        confidence >= self.lower_conf_threshold
                        and calculate_iou(tracking_bbox, detection_bbox)
                        >= self.iou_threshold
                    ):
                        if not np.any(
                            detection_index == np.array(self.used_detections_list)
                        ):
                            matched_detection_index = detection_index
                            self.used_detections_list.append(detection_index)
                            break
            track.detection_index_list.append(matched_detection_index)
        return track

    def _save_track(self, track):

        n_bbox = len(track.bbox_list)
        if n_bbox < self.min_length:
            return

        for index, detection_index in enumerate(track.detection_index_list):

            if detection_index != -1:
                class_id = self.detection_class_list[detection_index]
                confidence = self.detection_confidence_list[detection_index]
            else:
                class_id = str(detection_index)
                confidence = detection_index
            track.detection_class_list.append(class_id)
            track.detection_score_list.append(confidence)

        if self.run_classifier:
            class_list, score_list = track._classify(
                self.input_data, self.classifier, self.classification_window
            )
            track.classification_list = class_list
            track.classification_score_list = score_list

        if self.save_crop_images:
            crop_path = os.path.join(self.output_path, str(track.track_id))
            os.makedirs(crop_path, exist_ok=True)
            track._save_crops(crop_path, self.input_data)

        track._save_json_output(
            self.output_path, track.track_id, self.img_width, self.img_height
        )

    def _get_bbox_wh(self, bbox):
        w = bbox[3] - bbox[1]
        h = bbox[2] - bbox[0]
        return w, h

    def _check_if_bbox_too_small(self, bbox):
        # Check if bounding box is too small
        w, h = self._get_bbox_wh(bbox)
        if w * h < self.min_bbox_area:
            return True
        else:
            return False

    def _end_track_check(self, track):
        # Check if tracker is still detecting the object
        centroid_list = []
        bbox_sizes_list = []
        displacement_list = []
        displacement_length_list = []
        counter = 0
        for t_step, bbox in enumerate(track.bbox_list):
            # Check if tracker returned multiple non-detections
            if track.tracking_flag_list[t_step] is not None:
                counter += 1
                if counter >= self.end_threshold:
                    t_step = t_step - self.end_threshold
                    track.is_end_track = True
                    break
            else:
                counter = 0

            # Check if bounding box is too small
            if self._check_if_bbox_too_small(bbox):
                track.is_end_track = True
                break

            # Check if object is moving relative to camera
            w, h = self._get_bbox_wh(bbox)
            bbox_sizes_list.append([h, w])
            centroid_list.append([bbox[0] + h // 2, bbox[1] + w // 2])

            if len(centroid_list) >= self.max_length:
                track.is_end_track = True
                break

            if len(centroid_list) >= 2:
                displacement_list.append(
                    np.array(centroid_list[t_step])
                    - np.array(centroid_list[t_step - 1])
                )
                displacement_length_list.append(np.linalg.norm(displacement_list[-1]))
                if len(centroid_list) >= self.end_threshold and np.all(
                    np.array(displacement_length_list)[-self.end_threshold :]
                    < self.min_displacement
                ):
                    t_step = t_step - self.end_threshold
                    track.is_end_track = True
                    break
                if len(centroid_list) >= self.min_length:
                    # Check if object trajectory goes out-of-frame
                    try:  # Use auto-regressive model
                        centroid_pred = predict_trajectory(centroid_list[:-1])
                    except:  # Just use last displacement
                        centroid_pred = centroid_list[-2] + displacement_list[-2]
                    h_max, w_max = np.max(
                        np.array(bbox_sizes_list), axis=0
                    )  # Take the largest bounding boxes from the object track (which should be the closest one)
                    if (
                        centroid_pred[0] - h_max // 2 < self.crop_window[0]
                        or centroid_pred[0] + h_max // 2
                        >= self.img_height - self.crop_window[1]
                        or centroid_pred[1] - w_max // 2 < self.crop_window[2]
                        or centroid_pred[1] + w_max // 2
                        >= self.img_width - self.crop_window[3]
                    ):
                        track.is_end_track = True
                        break
                    # Check if object trajectory is inconsistent
                    if self.max_relative_diff > 0:
                        centroid_last = centroid_list[-1]
                        abs_diff_y = np.abs(centroid_last[0] - centroid_pred[0])
                        abs_diff_x = np.abs(centroid_last[1] - centroid_pred[1])
                        norm_y = np.abs(displacement_list[-1][0])
                        norm_x = np.abs(displacement_list[-1][1])
                        rel_diff_y = abs_diff_y / norm_y
                        rel_diff_x = abs_diff_x / norm_x
                        if (
                            rel_diff_x > self.max_relative_diff
                            or rel_diff_y > self.max_relative_diff
                        ):
                            track.is_end_track = True
                            break

        if track.is_end_track:
            if t_step < 0:
                t_step = 0
            track.last_frame_id = track.frame_id_list[t_step]

        return track

    def _get_track(self, track, direction="forwards"):
        while not track.is_end_track:
            track = self._get_tracking_prediction(track, direction=direction)
            track = self._match_detections(track)
            if self.use_detection:
                for index, detection_index in enumerate(track.detection_index_list):
                    if detection_index != -1:
                        track.bbox_list.append(
                            self.detection_bbox_list[detection_index]
                        )
                        track.is_detected_list.append(True)
                    else:
                        track.bbox_list.append(self.tracking_bbox_list[index])
                        track.is_detected_list.append(False)
            else:
                track.bbox_list = track.tracking_bbox_list.copy()
                track.is_detected_list = [False for i in range(len(track.bbox_list))]
                track.is_detected_list[0] = True  # Initial detection bounding box
            track.last_frame_id = track.frame_id_list[-1]
            track.init_bbox = track.bbox_list[-1]
            track = self._end_track_check(track)
        track._slice_track()

    def tracking(self, input_data, detection_data, output_path):
        self.detection_data = detection_data
        self.input_data = input_data
        self.output_path = output_path
        os.makedirs(self.output_path, exist_ok=True)
        # Get detections
        self.detection_bbox_list = []
        self.detection_frame_id_list = []
        self.detection_class_list = []
        self.detection_confidence_list = []
        for detection_index, detection_path in enumerate(self.detection_data):
            # load the detections JSON
            detection_list_for_frame = read_detection_file_json(detection_path)
            detection_list_for_frame = np.array(detection_list_for_frame, dtype=object)
            filename = os.path.splitext(os.path.basename(detection_path))[0]
            filename_split = filename.split("_")
            frame_id = filename_split[-1]
            if frame_id.isdigit():
                frame_id = int(frame_id)
            else:
                frame_id = detection_index
            for detection in detection_list_for_frame:
                bbox = detection[:4]
                if not self._check_if_bbox_too_small(bbox):
                    self.detection_bbox_list.append(bbox)
                    self.detection_class_list.append(detection[-2])
                    self.detection_confidence_list.append(detection[-1])
                    self.detection_frame_id_list.append(int(frame_id))

        # Order detections
        sort_indices = np.flip(np.argsort(self.detection_confidence_list))

        self.detection_bbox_list = np.array(self.detection_bbox_list)
        self.detection_class_list = np.array(self.detection_class_list)
        self.detection_confidence_list = np.array(self.detection_confidence_list)
        self.detection_frame_id_list = np.array(self.detection_frame_id_list)

        self.detection_bbox_list = self.detection_bbox_list[sort_indices]
        self.detection_class_list = self.detection_class_list[sort_indices]
        self.detection_confidence_list = self.detection_confidence_list[sort_indices]
        self.detection_frame_id_list = self.detection_frame_id_list[sort_indices]

        # Start tracking
        track_id = 0
        self.used_detections_list = []
        for detection_index in tqdm(range(len(self.detection_frame_id_list))):
            detection_confidence = self.detection_confidence_list[detection_index]
            if detection_confidence >= self.upper_conf_threshold and not np.any(
                detection_index == np.array(self.used_detections_list)
            ):
                init_bbox = self.detection_bbox_list[detection_index]
                frame_id = self.detection_frame_id_list[detection_index]
                # Create new track
                track_forw = Track(init_bbox, frame_id, track_id)
                track_back = Track(init_bbox, frame_id, track_id)
                # Go forwards in time
                self._get_track(track_forw, direction="forwards")
                # Go backwards in time
                self._get_track(track_back, direction="backwards")
                # Concatenate track results
                track_forw._concatenate_tracks(track_back)
                self._save_track(track_forw)
                # Update track id
                track_id += 1


if __name__ == "__main__":

    from options import get_arguments

    ARGS = get_arguments()

    tracker = Tracker(
        classifier_path=ARGS.classifier_path,
        method=ARGS.method,
        input_type=ARGS.input_type,
        output_filename=ARGS.output_filename,
        num_future_frames=ARGS.num_future_frames,
        min_length=ARGS.min_length,
        max_length=ARGS.max_length,
        lower_conf_threshold=ARGS.lower_conf_threshold,
        upper_conf_threshold=ARGS.upper_conf_threshold,
        iou_threshold=ARGS.iou_threshold,
        max_relative_diff=ARGS.max_relative_diff,
        img_height=ARGS.img_height,
        img_width=ARGS.img_width,
        crop_window=ARGS.crop_window,
        min_bbox_area=ARGS.min_bbox_area,
        min_displacement=ARGS.min_displacement,
        save_crop_images=ARGS.save_crop_images,
        run_classifier=ARGS.run_classifier,
        use_detection=ARGS.use_detection,
        classifier_datatype=ARGS.classifier_datatype,
        classifier_datadir=ARGS.classifier_datadir,
        classifier_image_size=ARGS.classifier_image_size,
        end_threshold=ARGS.end_threshold,
        classification_window=ARGS.classification_window,
    )

    detec_file_list, input_file_list = get_detection_and_image_list(
        ARGS.detection_path, ARGS.input_path
    )
    tracker.tracking(input_file_list, detec_file_list, ARGS.output_path)
