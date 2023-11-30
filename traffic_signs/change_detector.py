# Copyright (C) 2020 by NavInfo Europe B.V. The Netherlands - All rights reserved
# Information classification: Confidential
# This content is protected by international copyright laws.
# Reproduction and distribution is prohibited without written permission.

import glob
import logging
import numpy as np
import os
import sys
import threading

from .change_detection import change_detection

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_PATH, ".."))
from utils import get_mapping_lists

sys.path.insert(
    0,
    os.path.join(
        SCRIPT_PATH,
        "..",
        "object_detection",
        "common",
        "libs",
        "nie_tracking",
        "python",
    ),
)
from tracker_utils import filter_tracks, read_input_file


def get_tracking_results(cache_dir_run, vanishing_points):

    results_filepath_list = glob.glob(
        os.path.join(cache_dir_run, "results_track_*.json")
    )
    results_filepath_list.sort()

    tracks_to_keep = filter_tracks(
        results_filepath_list, vanishing_points, do_filter_track_trajectory=False
    )
    results_filepath_list = np.array(results_filepath_list)
    results_filepath_list = list(results_filepath_list[tracks_to_keep])

    result_list = []
    for result_filepath in results_filepath_list:
        file_content_dict = read_input_file(result_filepath)
        frame_id_list = file_content_dict["frame_id_list"]
        result = [
            min(frame_id_list),
            max(frame_id_list),
            file_content_dict["class_id"],
            file_content_dict["score"],
            file_content_dict["instance_id"],
        ]
        result_list.append(result)
    result_list = np.array(result_list)
    if len(result_list) > 0:
        sort_indices = np.argsort(result_list[:, 0])
        result_list = result_list[sort_indices, :]
    return result_list


def traffic_sign_detection(detector, run_path_list, cache_dir, run="run1"):
    logging.info("Running detection on " + run + "...")
    output_dir = os.path.join(cache_dir, "detection", run)
    os.makedirs(output_dir, exist_ok=True)
    for image_path in run_path_list:
        detector.detect(image_path, output_dir)


def traffic_sign_tracking(tracker, run_path_list, cache_dir, run="run1"):
    logging.info("Running tracking on " + run + "...")
    detection_dir = os.path.join(cache_dir, "detection", run)
    output_dir = os.path.join(cache_dir, "tracking", run)
    os.makedirs(output_dir, exist_ok=True)
    tracker.track(run_path_list, detection_dir, output_dir)


def align_tracking_results(tracking_result_list, mapping_list):
    for index, tracking_result in enumerate(tracking_result_list):
        tracking_result_list[index, 0] = mapping_list[tracking_result[0]][0]
        tracking_result_list[index, 1] = mapping_list[tracking_result[1]][1]
    return tracking_result_list


def process_changes_list(
    changes_list, run1_coords, run2_coords, run1_path_list, run2_path_list
):
    """
    Function to put the changes in the final CSV format.
    NOTE: DOES NOT write the final CSV though
    :param changes_list: added/removed changes along with details
    :param run1_path_list and run2_path_list: the list of img paths from run1 and run2
    :return: list of lists formatted in a way so that it can be added to the CSV
    """

    processed_final_changes_list = list()

    for i, change in enumerate(changes_list, 1):
        major_category = "Traffic Sign/Light"
        object_status_change = change[0]
        minor_category = str(change[1])
        frame_start = int(change[2])
        frame_end = int(change[3])

        gps_run1 = run1_coords[frame_end]
        filename_run1 = run1_path_list[frame_end]
        gps_run2 = run2_coords[frame_end]
        filename_run2 = run2_path_list[frame_end]

        processed_final_changes_list.append(
            [
                object_status_change,
                major_category,
                minor_category,
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                gps_run1,
                gps_run2,
                "",
                "",
                "",
                "",
                filename_run1,
                filename_run2,
                frame_start,
                frame_end,
            ]
        )
    return processed_final_changes_list


def detect_traffic_sign_changes(
    run1_coords,
    run2_coords,
    run1_path_list,
    run2_path_list,
    detector,
    tracker,
    cache_dir,
    vanishing_points_run1,
    vanishing_points_run2,
):
    if not os.path.isabs(cache_dir):
        cache_dir_split = cache_dir.split(os.sep)
        cache_dir = os.path.join(*cache_dir_split[1:])
        cache_dir = os.path.join(SCRIPT_PATH, "..", cache_dir)

    (
        run1_path_list_unique,
        run2_path_list_unique,
        mapping_list_run1,
        mapping_list_run2,
    ) = get_mapping_lists(run1_path_list, run2_path_list, cache_dir)

    # Do detection
    thread_1 = threading.Thread(
        target=traffic_sign_detection,
        args=(detector, run1_path_list_unique, cache_dir, "run1"),
    )
    thread_2 = threading.Thread(
        target=traffic_sign_detection,
        args=(detector, run2_path_list_unique, cache_dir, "run2"),
    )
    thread_1.start()
    thread_2.start()
    thread_1.join()
    thread_2.join()

    # Do tracking
    tracking_cache_dir = os.path.join(cache_dir, "tracking")
    if not os.path.isdir(tracking_cache_dir):
        traffic_sign_tracking(tracker, run1_path_list_unique, cache_dir, "run1")
        traffic_sign_tracking(tracker, run2_path_list_unique, cache_dir, "run2")

    logging.info("Getting tracking results...")
    tracking_cache_dir_run1 = os.path.join(tracking_cache_dir, "run1")
    tracking_cache_dir_run2 = os.path.join(tracking_cache_dir, "run2")
    results_run1 = get_tracking_results(tracking_cache_dir_run1, vanishing_points_run1)
    results_run2 = get_tracking_results(tracking_cache_dir_run2, vanishing_points_run2)

    if len(results_run1) == 0 or len(results_run2) == 0:
        logging.info(
            "No tracking results found for one of the runs. This indicates an error. Please remove 'tracking' folder and try again."
        )
        processed_changes_list = []
    else:
        logging.info("Align tracking results...")
        results_run1 = align_tracking_results(results_run1, mapping_list_run1)
        results_run2 = align_tracking_results(results_run2, mapping_list_run2)

        logging.info("Finding changes between run1 and run2...")
        changes_list = change_detection(
            results_run1=results_run1, results_run2=results_run2
        )

        logging.info("Processing changes...")
        processed_changes_list = process_changes_list(
            changes_list=changes_list,
            run1_coords=run1_coords,
            run2_coords=run2_coords,
            run1_path_list=run1_path_list,
            run2_path_list=run2_path_list,
        )

    return processed_changes_list
