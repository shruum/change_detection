# Copyright (C) 2020 by NavInfo Europe B.V. The Netherlands - All rights reserved
# Information classification: Confidential
# This content is protected by international copyright laws.
# Reproduction and distribution is prohibited without written permission.

import itertools
import numpy as np
import os
import sys

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))
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

CLASS_IGNORE_LIST = np.array(["-1"])


def add_changes(object_list, change_type):
    changes_list = []
    for obj in object_list:
        class_name = str(obj[0])
        frame_start = int(obj[1])
        frame_end = int(obj[2])
        change_sublist = [change_type, class_name, frame_start, frame_end]
        changes_list.append(change_sublist)
    return changes_list


def create_final_changes(object_list_added, object_list_removed):

    changes_list = []
    changes_list += add_changes(object_list=object_list_added, change_type="added",)
    changes_list += add_changes(object_list=object_list_removed, change_type="removed",)

    # sort the final changes list
    changes_list.sort(key=lambda x: x[1])

    return changes_list


def store_changes(
    frame_start_list,
    frame_end_list,
    label_list,
    score_list,
    matched_list,
    score_threshold=0.50,
):
    object_list_changes = []
    for object_index in range(len(frame_start_list)):
        if not np.any(object_index == matched_list):
            label_object_list = np.array(label_list[object_index])
            score_object_list = np.array(score_list[object_index])
            keep_indices = []
            for index, label_object in enumerate(label_object_list):
                if not np.any(CLASS_IGNORE_LIST == label_object):
                    keep_indices.append(index)
            if len(keep_indices) > 0:
                score_object_list = score_object_list[keep_indices]
                max_score_index = np.argmax(score_object_list)
                if score_object_list[max_score_index] >= score_threshold:
                    label = str(label_object_list[keep_indices[max_score_index]])
                    object_list_changes.append(
                        [
                            label,
                            frame_start_list[object_index],
                            frame_end_list[object_index],
                        ]
                    )
    return np.array(object_list_changes)


def calculate_frame_iou(
    frame_start_run1, frame_start_run2, frame_end_run1, frame_end_run2
):
    frame_start_max = np.max([frame_start_run1, frame_start_run2])
    frame_start_min = np.min([frame_start_run1, frame_start_run2])
    frame_end_max = np.max([frame_end_run1, frame_end_run2])
    frame_end_min = np.min([frame_end_run1, frame_end_run2])
    union = frame_end_max - frame_start_min
    intersection = frame_end_min - frame_start_max
    if union > 0 and intersection > 0:
        overlap = intersection / union
    else:
        overlap = 0
    return overlap


def find_changes(results_run1, results_run2, max_combinations=7):

    frame_range = [
        min(results_run1[:, 0]),
        min(results_run2[:, 0]),
        max(results_run1[:, 1]),
        max(results_run2[:, 1]),
    ]

    min_frame, max_frame = min(frame_range), max(frame_range)

    frame_start_list_run1 = np.array(results_run1[:, 0])
    frame_start_list_run2 = np.array(results_run2[:, 0])

    frame_end_list_run1 = np.array(results_run1[:, 1])
    frame_end_list_run2 = np.array(results_run2[:, 1])

    label_list_run1 = np.array(results_run1[:, 2])
    label_list_run2 = np.array(results_run2[:, 2])

    score_list_run1 = np.array(results_run1[:, 3])
    score_list_run2 = np.array(results_run2[:, 3])

    n_entries_run1 = len(frame_start_list_run1)
    n_entries_run2 = len(frame_start_list_run2)

    matching_matrix = np.zeros((n_entries_run1, n_entries_run2))

    for object_index_run1 in range(n_entries_run1):
        label_list_run1_frame = np.array(label_list_run1[object_index_run1])
        for object_index_run2 in range(n_entries_run2):
            label_list_run2_frame = np.array(label_list_run2[object_index_run2])
            comparisons = np.in1d(label_list_run1_frame, label_list_run2_frame)
            if np.any(comparisons):
                frame_start_run1 = frame_start_list_run1[object_index_run1]
                frame_start_run2 = frame_start_list_run2[object_index_run2]
                frame_end_run1 = frame_end_list_run1[object_index_run1]
                frame_end_run2 = frame_end_list_run2[object_index_run2]
                iou = calculate_frame_iou(
                    frame_start_run1, frame_start_run2, frame_end_run1, frame_end_run2,
                )
                matching_matrix[object_index_run1, object_index_run2] = iou

    overall_match_list = []
    while True:
        best_match = np.unravel_index(matching_matrix.argmax(), matching_matrix.shape)
        iou = matching_matrix[best_match]
        if iou == 0:
            break
        iou_threshold_list = np.linspace(0, 1.0, 11)
        for current_iou_threshold in iou_threshold_list:
            current_match_list = []
            current_matching_matrix = matching_matrix.copy()
            current_matching_matrix[best_match] = 0
            iou_list = []
            iou_list.append(iou)
            # Find all other matches
            active_match_list = []
            active_match_list.append(np.array(best_match))
            while len(active_match_list) > 0:
                active_match_list_new = []
                for match in active_match_list:
                    x = match[0]
                    y = match[1]
                    # Save match
                    current_match_list.append(match)
                    # Find all other matches
                    x_indices = np.nonzero(
                        current_matching_matrix[:, match[1]] > current_iou_threshold
                    )[0]
                    y_indices = np.nonzero(
                        current_matching_matrix[match[0], :] > current_iou_threshold
                    )[0]
                    for x_index in x_indices:
                        active_match_list_new.append(np.array([x_index, y]))
                        iou_list.append(current_matching_matrix[x_index, y])
                        current_matching_matrix[x_index, y] = 0
                    for y_index in y_indices:
                        active_match_list_new.append(np.array([x, y_index]))
                        iou_list.append(current_matching_matrix[x, y_index])
                        current_matching_matrix[x, y_index] = 0
                active_match_list = active_match_list_new
            current_match_list = np.array(current_match_list)
            if len(current_match_list) <= max_combinations:
                break
        if len(current_match_list) > max_combinations:
            sort_indices = np.argsort(np.array(iou_list))
            current_match_list = current_match_list[sort_indices]
            current_match_list = current_match_list[:max_combinations]
        permutations_list = list(
            itertools.permutations(np.arange(0, len(current_match_list)))
        )
        n_matches_list = []
        iou_list = []

        def find_matches_for_permutation(current_match_list, permutation):
            match_list = []
            for i in permutation:
                x, y = current_match_list[i, :]
                if x != -1 and y != -1:
                    current_match_list[current_match_list == x] = -1
                    current_match_list[current_match_list == y] = -1
                    match_list.append(np.array([x, y]))
                if np.all(current_match_list) == -1:
                    break
            return match_list

        for permutation in permutations_list:
            match_list = find_matches_for_permutation(
                current_match_list.copy(), permutation
            )
            n_matches = len(match_list)
            iou = 0
            for match in match_list:
                iou += matching_matrix[match[0], match[1]]
            iou_list.append(iou)
            n_matches_list.append(n_matches)
            if n_matches == len(permutation):
                break
        n_matches_list = np.array(n_matches_list)
        max_n_matches = np.max(n_matches_list)
        indices_matches = np.nonzero(n_matches_list == max_n_matches)[0]
        iou_list = np.array(iou_list)
        max_iou_index = np.argmax(iou_list[indices_matches])
        best_permutation_index = indices_matches[max_iou_index]
        best_permutation = permutations_list[best_permutation_index]
        match_list = find_matches_for_permutation(
            current_match_list.copy(), best_permutation
        )
        for match in match_list:
            matching_matrix[match[0], :] = 0
            matching_matrix[:, match[1]] = 0
        overall_match_list.append(match_list)
    overall_match_list = np.vstack(overall_match_list)

    object_list_removed = store_changes(
        frame_start_list_run1,
        frame_end_list_run1,
        label_list_run1,
        score_list_run1,
        overall_match_list[:, 0],
    )
    object_list_added = store_changes(
        frame_start_list_run2,
        frame_end_list_run2,
        label_list_run2,
        score_list_run2,
        overall_match_list[:, 1],
    )

    return object_list_added, object_list_removed, min_frame, max_frame


def change_detection(results_run1, results_run2):
    object_list_added, object_list_removed, min_frame, max_frame = find_changes(
        results_run1, results_run2
    )
    changes_list = create_final_changes(object_list_added, object_list_removed)
    return changes_list
