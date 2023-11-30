import collections
import glob
import json
import numpy as np
import os
from statsmodels.tsa.vector_ar.var_model import VAR

Rectangle = collections.namedtuple("Rectangle", ["x", "y", "width", "height"])


def get_detection_and_image_list(detection_dir, image_dir):
    # Get image and detection files
    detection_data = glob.glob(os.path.join(detection_dir, "*.json"))
    detection_data.sort()
    detec_file_list = []
    input_file_list = []
    for detec_file in detection_data:
        basename_detec = os.path.splitext(
            os.path.basename(os.path.normpath(detec_file))
        )[0]
        input_file = os.path.join(image_dir, basename_detec)
        input_file_png = input_file + ".png"
        input_file_jpg = input_file + ".jpg"
        if os.path.isfile(input_file_png):
            detec_file_list.append(detec_file)
            input_file_list.append(input_file_png)
        elif os.path.isfile(input_file_jpg):
            detec_file_list.append(detec_file)
            input_file_list.append(input_file_jpg)
    return detec_file_list, input_file_list


def calculate_intersection(box1, box2):
    if box1[3] < box2[1] or box1[1] > box2[3] or box1[2] < box2[0] or box1[0] > box2[2]:
        return 0.0, 0.0, 0.0
    ixmin = max(box1[1], box2[1])
    iymin = max(box1[0], box2[0])
    ixmax = min(box1[3], box2[3])
    iymax = min(box1[2], box2[2])
    bbox_area_1 = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    bbox_area_2 = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    iw = ixmax - ixmin + 1
    ih = iymax - iymin + 1
    ia = float(iw * ih)
    return ia, bbox_area_1, bbox_area_2


def calculate_iou(box1, box2):
    intersection, bbox_area_1, bbox_area_2 = calculate_intersection(box1, box2)
    union = bbox_area_1 + bbox_area_2 - intersection
    if union > 0.0:
        return float(intersection) / float(union)
    else:
        return 0.0


def write_track_keep(filepath, keep, overwrite=False):
    with open(filepath) as file:
        track_file_dict = json.load(file)
    if overwrite or ("keep" not in track_file_dict):
        track_file_dict["keep"] = bool(keep)
        with open(filepath, "w", encoding="utf-8") as file:
            json.dump(track_file_dict, file, ensure_ascii=False, indent=3)


def write_track_class_and_score(filepath, class_list, score_list, overwrite=False):
    with open(filepath) as file:
        track_file_dict = json.load(file)
    if overwrite or (
        "class_id" not in track_file_dict or "score" not in track_file_dict
    ):
        track_file_dict["class_id"] = class_list
        track_file_dict["score"] = score_list
        with open(filepath, "w", encoding="utf-8") as file:
            json.dump(track_file_dict, file, ensure_ascii=False, indent=3)


def read_input_file(input_filepath):
    with open(input_filepath) as file:
        track_file_dict = json.load(file)
    track_object_list = track_file_dict["objects"]
    file_content_dict = {}
    file_content_dict["frame_id_list"] = []
    file_content_dict["bbox_list"] = []
    file_content_dict["class_id_list"] = []
    file_content_dict["confidence_list"] = []
    for track_object in track_object_list:
        file_content_dict["frame_id_list"].append(track_object["frame_id"])
        file_content_dict["bbox_list"].append(
            [
                track_object["x0"],
                track_object["y0"],
                track_object["x1"],
                track_object["y1"],
            ]
        )
        file_content_dict["class_id_list"].append(track_object["classification_list"])
        file_content_dict["confidence_list"].append(
            track_object["classification_score_list"]
        )
    file_content_dict["instance_id"] = track_file_dict["instance_id"]
    if "class_id" in track_file_dict:
        file_content_dict["class_id"] = track_file_dict["class_id"]
    if "score" in track_file_dict:
        file_content_dict["score"] = track_file_dict["score"]
    if "keep" in track_file_dict:
        file_content_dict["keep"] = track_file_dict["keep"]
    return file_content_dict


def get_confidence_score(file_content_dict):
    class_id_list = np.array(file_content_dict["class_id_list"]).flatten()
    confidence_list = np.array(file_content_dict["confidence_list"]).flatten()
    # Ignore confidences of -1
    keep = confidence_list != -1
    class_id_list = class_id_list[keep]
    confidence_list = confidence_list[keep]
    # Get overall score for each class
    n_frames = len(class_id_list) // 3
    unique_classes = np.unique(class_id_list)
    scores = []
    for class_id in unique_classes:
        confidences = confidence_list[class_id == class_id_list]
        confidences = [float(x) for x in confidences]
        scores.append(np.sum(confidences) / n_frames)
    scores = np.array(scores)
    return scores, unique_classes


def filter_track_confidence_score(results_filepath_list, score_threshold):
    tracks_to_delete = []
    for track_index, results_filepath in enumerate(results_filepath_list):
        file_content_dict = read_input_file(results_filepath)
        scores, unique_classes = get_confidence_score(file_content_dict)
        indices_to_keep = scores > score_threshold
        if np.any(indices_to_keep):
            class_list = unique_classes[indices_to_keep]
            score_list = scores[indices_to_keep]
            sort_indices = np.argsort(score_list)
            class_list = class_list[sort_indices].tolist()
            score_list = score_list[sort_indices].tolist()
        else:
            tracks_to_delete.append(track_index)
        write_track_class_and_score(results_filepath, class_list, score_list)
    return tracks_to_delete


def filter_track_trajectory(
    results_filepath_list, vanishing_points, diff_threshold, smoothing_window_size
):
    tracks_to_delete = []
    for track_index, results_filepath in enumerate(results_filepath_list):
        file_content_dict = read_input_file(results_filepath)
        # Make sure trajectory is sensible
        vpoint_x = vanishing_points[0]
        vpoint_y = vanishing_points[1]
        centroid_x_list = []
        centroid_y_list = []
        for bbox in file_content_dict["bbox_list"]:
            x = bbox[0] + (bbox[2] - bbox[0]) / 2
            y = bbox[1] + (bbox[3] - bbox[1]) / 2
            centroid_x_list.append(x)
            centroid_y_list.append(y)
        centroid_x_list = np.array(centroid_x_list)
        centroid_y_list = np.array(centroid_y_list)
        # Convolve to account for jittering
        sliding_window = np.ones((smoothing_window_size,)) / smoothing_window_size
        centroid_x_list_smoothed = np.convolve(
            centroid_x_list, sliding_window, mode="valid"
        )
        centroid_y_list_smoothed = np.convolve(
            centroid_y_list, sliding_window, mode="valid"
        )
        vector_difference_list = []
        vector_sum = 0
        for index in range(len(centroid_x_list_smoothed) - 1):
            x = centroid_x_list_smoothed[index]
            y = centroid_y_list_smoothed[index]
            dx = centroid_x_list_smoothed[index + 1] - x
            dy = centroid_y_list_smoothed[index + 1] - y
            vector = np.array([dx, dy])
            expected_vector = -np.array([vpoint_x - x, vpoint_y - y])
            vector_length = np.linalg.norm(vector)
            expected_vector_length = np.linalg.norm(expected_vector)
            if vector_length > 0 and expected_vector_length > 0:
                unit_vector = vector / vector_length
                expected_unit_vector = expected_vector / expected_vector_length
                vector_difference_length = vector_length * np.linalg.norm(
                    unit_vector - expected_unit_vector
                )
                vector_sum += vector_length
                vector_difference_list.append(vector_difference_length)
        if len(vector_difference_list) > 0:
            vector_difference = np.sum(np.array(vector_difference_list)) / vector_sum
            if vector_difference > diff_threshold:
                tracks_to_delete.append(track_index)
    return tracks_to_delete


def filter_track_overlap(results_filepath_list, iou_threshold=0.25):
    # Sort tracks based on end frame id
    max_frame_id_list = []
    for results_filepath in results_filepath_list:
        file_content_dict = read_input_file(results_filepath)
        max_frame = file_content_dict["frame_id_list"][-1]
        max_frame_id_list.append(max_frame)
    max_frame_id_list = np.array(max_frame_id_list)
    sort_indices = np.argsort(max_frame_id_list)
    results_filepath_list = np.array(results_filepath_list)
    results_filepath_list_sorted = results_filepath_list[sort_indices]
    # Check overlap between tracks, remove smallest bounding box track
    tracks_to_delete = []
    n_tracks = len(results_filepath_list_sorted)
    for track_index_1, results_filepath_1 in enumerate(results_filepath_list_sorted):
        file_content_dict_1 = read_input_file(results_filepath_1)
        min_frame_1 = file_content_dict_1["frame_id_list"][0]
        max_frame_1 = file_content_dict_1["frame_id_list"][-1]
        class_id_array_1 = np.array(file_content_dict_1["class_id"])
        no_overlap_counter = 0
        for track_index_2 in range(track_index_1 + 1, n_tracks):
            results_filepath_2 = results_filepath_list_sorted[track_index_2]
            file_content_dict_2 = read_input_file(results_filepath_2)
            min_frame_2 = file_content_dict_2["frame_id_list"][0]
            max_frame_2 = file_content_dict_2["frame_id_list"][-1]
            class_id_array_2 = np.array(file_content_dict_2["class_id"])
            comparisons = np.in1d(class_id_array_1, class_id_array_2)
            if np.any(comparisons):
                if max_frame_1 >= min_frame_2 and min_frame_1 <= max_frame_2:
                    iou_list = []
                    for frame_index_1, frame_id_1 in enumerate(
                        file_content_dict_1["frame_id_list"]
                    ):
                        indices = np.nonzero(
                            frame_id_1 == np.array(file_content_dict_2["frame_id_list"])
                        )[0]
                        if len(indices) > 0:
                            frame_index_2 = indices[0]
                            bbox_1 = file_content_dict_1["bbox_list"][frame_index_1]
                            bbox_2 = file_content_dict_2["bbox_list"][frame_index_2]
                            iou = calculate_iou(bbox_1, bbox_2)
                            iou_list.append(iou)
                    if len(iou_list) >= 3:
                        iou_mean = np.mean(np.array(iou_list))
                        if iou_mean >= iou_threshold:
                            score_1 = np.max(file_content_dict_1["score"])
                            score_2 = np.max(file_content_dict_2["score"])
                            if score_1 <= score_2:
                                tracks_to_delete.append(track_index_1)
                            else:
                                tracks_to_delete.append(track_index_2)
                    no_overlap_counter = 0
                else:
                    no_overlap_counter += 1
            if no_overlap_counter >= 10:
                break
    # Convert to original track indices
    tracks_to_delete = sort_indices[tracks_to_delete]
    return tracks_to_delete


def filter_tracks(
    results_filepath_list,
    vanishing_points,
    score_threshold=0.20,
    diff_threshold=0.75,
    iou_threshold=0.50,
    smoothing_window_size=10,
    do_filter_confidence=True,
    do_filter_track_overlap=True,
    do_filter_track_trajectory=True,
):

    results_filepath_list = np.array(results_filepath_list)
    tracks_to_keep_mask = np.ones(len(results_filepath_list), dtype=bool)

    if do_filter_confidence:
        tracks_to_delete = filter_track_confidence_score(
            results_filepath_list, score_threshold
        )
        tracks_to_keep_mask[tracks_to_delete] = False

    tracks_to_keep_indices = np.nonzero(tracks_to_keep_mask)[0]
    results_filepath_list_kept = results_filepath_list[tracks_to_keep_indices]

    if do_filter_track_overlap:
        tracks_to_delete = filter_track_overlap(
            results_filepath_list_kept, iou_threshold
        )
        tracks_to_delete = tracks_to_keep_indices[tracks_to_delete]
        tracks_to_keep_mask[tracks_to_delete] = False
        tracks_to_keep_indices = np.nonzero(tracks_to_keep_mask)[0]

    results_filepath_list_kept = results_filepath_list[tracks_to_keep_indices]

    if do_filter_track_trajectory:
        tracks_to_delete = filter_track_trajectory(
            results_filepath_list_kept,
            vanishing_points,
            diff_threshold,
            smoothing_window_size,
        )
        tracks_to_delete = tracks_to_keep_indices[tracks_to_delete]
        tracks_to_keep_mask[tracks_to_delete] = False

    for track_index, results_filepath in enumerate(results_filepath_list):
        write_track_keep(results_filepath, tracks_to_keep_mask[track_index])

    return tracks_to_keep_mask


def group_junk_classes(class_id_list, score_list):
    score_list_combined = []
    junk_classes = np.array(["-1", "v", "p", "b", "lp", "hlm"])
    for class_index, class_id in enumerate(class_id_list):
        if np.any(junk_classes == class_id):
            class_id_list[class_index] = "-1"
    class_id_list = np.array(class_id_list)
    score_list = np.array(score_list)
    unique_class_id_list, unique_indices = np.unique(class_id_list, return_inverse=True)
    for index in range(len(unique_class_id_list)):
        score_list_combined.append(np.sum(score_list[unique_indices == index]))
    score_list_combined = np.array(score_list_combined)
    return unique_class_id_list, score_list_combined


def predict_trajectory(traj):
    model = VAR(traj)
    model_fit = model.fit()
    y_hat = model_fit.forecast(model_fit.y, steps=1)
    return y_hat[0]


def read_detection_file_json(detect_file):
    detections = []
    with open(detect_file) as json_file:
        data = json.load(json_file)
        for obj in data["objects"]:
            det_class = obj["f_name"]
            det_score = obj["f_conf"]
            x1 = obj["obj_points"]["x"]
            y1 = obj["obj_points"]["y"]
            x2 = x1 + obj["obj_points"]["w"]
            y2 = y1 + obj["obj_points"]["h"]
            det = [int(y1), int(x1), int(y2), int(x2), det_class, float(det_score)]
            detections.append(det)
    return detections


def convert_output_pytracking(output):
    trajectory_converted = []
    for bbox in output["target_bbox"]:
        trajectory_converted.append(Rectangle(bbox[0], bbox[1], bbox[2], bbox[3]))
    return trajectory_converted
