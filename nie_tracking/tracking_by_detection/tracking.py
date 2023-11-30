import argparse
from collections import Counter
import glob
import numpy as np
import os
import sys
import time
from tqdm import tqdm

from class_dictionary import CLASS_DICTIONARY

# Initialize tracker
CURRENT_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(CURRENT_DIR, "kiout"))
import iou_tracker
import util

sys.path.append(os.path.join(CURRENT_DIR, "kalman_filter_multi_object_tracking"))
from tracker import Tracker

sys.path.append(os.path.join(CURRENT_DIR, "..", "..", "data_processing", "apps"))
import data_processing_utils as utils


def get_unique_ids(detection_list):
    unique_id_list = np.unique(np.array(detection_list)[:, 1])
    unique_id_list = [int(x) for x in unique_id_list]
    return unique_id_list


def get_data_for_id(detection_list, instance_id_list, class_id):
    detection_list = np.array(detection_list)
    instance_id_list = np.array(instance_id_list)
    keep = np.array(detection_list)[:, 1] == class_id
    detection_list = list(detection_list[keep, :])
    instance_id_list = list(instance_id_list[keep])
    return detection_list, instance_id_list


def tracking(
    filepath_list, tracker_type, dataset_name, output_dir, distance_threshold=1000
):
    tracker_list = []
    global_instance_id = 0
    class_dict = CLASS_DICTIONARY[dataset_name]
    filepath_list = sorted(filepath_list)
    for frame_id in tqdm(range(len(filepath_list))):
        filepath = filepath_list[frame_id]
        if frame_id == 0 or tracker_type == "kfmot":
            detection_list_converted = []
            instance_id_list = []
            class_name_list = []
        detection_list = utils.get_detections_from_file(filepath)
        for detection in detection_list:
            y0, x0, h, w, class_name, _, confidence = detection
            class_id = class_dict[class_name]
            detection_converted = [
                frame_id,
                class_id,
                x0,
                y0,
                w,
                h,
                confidence,
            ]
            detection_list_converted.append(detection_converted)
            instance_id_list.append(global_instance_id)
            class_name_list.append(class_name)
            global_instance_id += 1
        if tracker_type == "kfmot":
            unique_class_id_list = get_unique_ids(detection_list_converted)
            for unique_class_id in unique_class_id_list:
                detection_list_class, instance_id_list_class = get_data_for_id(
                    detection_list_converted, instance_id_list, unique_class_id
                )
                while unique_class_id >= len(tracker_list):
                    tracker_list.append(Tracker(distance_threshold, 30, 100, 0))
                if len(instance_id_list_class) > 0:
                    tracker_list[unique_class_id].Update(
                        detection_list_class, instance_id_list_class, frame_id
                    )
    if tracker_type == "kiout":
        unique_class_id_list = get_unique_ids(detection_list_converted)
        dt_list = []
        for unique_class_id in unique_class_id_list:
            detection_list_class, instance_id_list_class = get_data_for_id(
                detection_list_converted, instance_id_list, unique_class_id
            )
            while unique_class_id >= len(tracker_list):
                tracker_list.append([])
            if len(detection_list_class) > 0:
                detection_list = util.load_mot(
                    np.array(detection_list_class), np.array(instance_id_list_class)
                )
                t0 = time.time()
                tracker_list[unique_class_id] = iou_tracker.track_iou(
                    detection_list, sigma_l=0.3, sigma_iou=0.2, sigma_p=16, sigma_len=3
                )
                t1 = time.time()
                dt = t1 - t0
                dt_list.append(dt)
        dt = np.sum(dt_list)
        print("Tracking took:", str(dt), "seconds")
        print("Corresponding to:", str(len(filepath_list) / dt), "FPS")

    convert_to_xml(
        tracker_list,
        filepath_list,
        detection_list_converted,
        class_name_list,
        output_dir,
        tracker_type="kiout",
    )


def convert_to_xml(
    tracker_list,
    filepath_list,
    detection_list_converted,
    class_name_list,
    output_dir,
    tracker_type="kiout",
):
    if tracker_type == "kiout":
        detections_to_add = []
        frame_id_list = []
        for tracker_class_id in tracker_list:
            for tracker_object in tracker_class_id:
                instance_id_list_old = []
                track_instance_list = []
                for track_instance in tracker_object:
                    id = track_instance["id"]
                    if id >= 0:
                        instance_id_list_old.append(id)
                    else:
                        track_instance_list.append(track_instance)
                instance_id_list_old = np.array(instance_id_list_old)
                class_name_list_object = np.array(class_name_list)[instance_id_list_old]
                confidence_list = np.array(detection_list_converted)[
                    instance_id_list_old
                ][:, -1]
                c = Counter(class_name_list_object)
                class_name = str(c.most_common(1)[0][0])
                confidence = np.mean(confidence_list)
                for track_instance in track_instance_list:
                    y0, x0, y1, x1 = track_instance["roi"]
                    x0, y0, w, h = utils.xyxy_to_xywh([x0, y0, x1, y1])
                    detections_to_add.append(
                        [y0, x0, h, w, class_name, None, confidence]
                    )
                    frame_id_list.append(track_instance["frame"])
        frame_id_list = np.array(frame_id_list)
        detections_to_add = np.array(detections_to_add)
        for frame_id, filepath in enumerate(filepath_list):
            annotation_dict = utils.load_xml_annotation_dict(filepath)
            detection_list = utils.get_detections_from_xml_dict(annotation_dict)
            img_width = annotation_dict["size"]["width"]
            img_height = annotation_dict["size"]["height"]
            detections_to_add_frame = detections_to_add[frame_id_list == frame_id]
            detection_list = detection_list + list(detections_to_add_frame)
            xml_dict = utils.convert_to_xml(
                detection_list, annotation_dict["filename"], img_width, img_height
            )
            output_path = os.path.join(output_dir, os.path.basename(filepath))
            utils.save_xml(xml_dict, output_path)


def convert_output(tracker_list, metadata, tracker_type="non_iou"):
    output_data = []
    object_id = 0
    for class_id, track_class_id in enumerate(tracker_list):
        if tracker_type == "kfmot":
            for track in track_class_id.tracks:
                object_track = []
                for i, instance_id in enumerate(track.trace_ids):
                    if instance_id != -1:
                        polygon = get_polygon(track, metadata, i)
                        X, Y = polygon.exterior.coords.xy
                        x = [int(x) for x in X]
                        y = [int(y) for y in Y]
                        frame_id = track.frame_ids[i]
                        object_track.append([frame_id, x, y])
                output_data.append([object_id, class_id, object_track])
                object_id += 1
        elif tracker_type == "kiout":
            for track in track_class_id:
                object_track = []
                for track_frame in track:
                    roi = track_frame["roi"]
                    x = roi[1::2]
                    y = roi[0::2]
                    x = [x[0], x[1], x[1], x[0]]
                    y = [y[0], y[0], y[1], y[1]]
                    frame_id = track_frame["frame"]
                    object_track.append([frame_id, x, y])
                if len(object_track) > 1:
                    output_data.append([object_id, class_id, object_track])
                object_id += 1
    return output_data


def load_images(data_dir, frame_offset, identifier="processed"):
    image_path_list = sorted(glob.glob(os.path.join(data_dir, "panoramic_*.jpg")))
    label_path_list = []
    frame_id_list = []
    for frame_id, image_path in enumerate(image_path_list):
        basename = os.path.splitext(os.path.basename(image_path))[0]
        basename_split = basename.split("_")
        frame_id = int(basename_split[-1])
        frame_id_list.append(frame_id)
        frame_id = frame_id + frame_offset
        frame_id = str(frame_id)
        label_path = os.path.join(
            os.path.dirname(image_path),
            basename_split[0] + "_" + identifier + "_" + frame_id + ".png",
        )
        if os.path.isfile(label_path):
            label_path_list.append(label_path)
    sort_indices = np.argsort(frame_id_list)
    image_path_list = np.array(image_path_list)[sort_indices]
    label_path_list = np.array(label_path_list)[sort_indices]
    return image_path_list, label_path_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/data/nie/teams/arl/projects/ceyebetron/tracker/test_videos/testclip1/detections_fixed/",
        help="Path to detection xml files",
    )
    parser.add_argument(
        "--tracker_type",
        type=str,
        default="kiout",
        help="Tracker type {'kiout' or 'kfmot'}",
    )
    parser.add_argument("--dataset_name", type=str, default="ark_22")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/data/nie/teams/arl/projects/ceyebetron/tracker/test_videos/testclip1/tracking/",
        help="Output dir",
    )
    args = parser.parse_args()

    xml_filepath_list = glob.glob(os.path.join(args.data_dir, "*.xml"))

    tracker_list = tracking(
        xml_filepath_list,
        tracker_type=args.tracker_type,
        dataset_name=args.dataset_name,
        output_dir=args.output_dir,
    )
