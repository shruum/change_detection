import numpy as np
from geopy.distance import geodesic
from scipy.signal import find_peaks
import textdistance
from geo_tag_CRD import write_dets_npy
import json
import os


def list_to_string(lst):
    strng = ""
    for val in lst:
        strng = strng.join(val)
    return strng


def get_detection_sets(info):
    if len(info.shape) < 1:
        print("No detections found in one of the two runs so nothing to compare.")
        return []
    cords = info[:, 1]
    offset_cords = cords[1:]
    cords = cords[:-1]
    dists = [
        geodesic((cord[0] - offset_cord[0]), (cord[1] - offset_cord[1])).m
        for cord, offset_cord in zip(cords, offset_cords)
    ]
    peaks, _ = find_peaks(dists, height=100)
    result = []
    prev_peak, peak = 0, 0
    # print(peaks)
    for peak in peaks:
        result.append(info[prev_peak:peak])
        prev_peak = peak
    result.append(info[peak + 1 :])
    return result


def pick_high_conf(values):
    conf_sum = None
    for value in values:
        flat_val = [item for sublist in value[-2] for item in sublist][0]
        if conf_sum is None:
            conf_sum = [sum(flat_val)]
        else:
            conf_sum.append(sum(flat_val))
    ind = np.argmax(conf_sum)
    processed_list = values[ind]
    processed_list[1] = values[-1, 1]
    return conf_sum, processed_list


def align_processed_batches(batches_1, batches_2):
    cords_1 = [batch[1] for batch in batches_1]
    cords_2 = [batch[1] for batch in batches_2]
    paired_batches = None
    n_list, idx_list = [], []
    for n, cord_1 in enumerate(cords_1):
        dists = [
            geodesic((cord_1[0] - cord_2[0]), (cord_1[1] - cord_2[1])).m
            for cord_2 in cords_2
        ]
        idx = np.argmin(dists)
        if dists[int(idx)] < 50:
            paired_batch = (batches_1[n], batches_2[idx])
            # x1, x2 = paired_batch[0][2], paired_batch[1][2]
            # print(paired_batch[0][2])
            # print(paired_batch[1][2])
            # cv2.imshow("1", np.hstack([img1, img2]))
            # cv2.waitKey()
            if paired_batches is None:
                paired_batches = [paired_batch]
            else:
                paired_batches.append(paired_batch)
            n_list.append(n)
            idx_list.append(idx)
    remaining_batches_1 = [
        batch for n, batch in enumerate(batches_1) if n not in n_list
    ]
    remaining_batches_2 = [
        batch for idx, batch in enumerate(batches_2) if idx not in idx_list
    ]

    return paired_batches, remaining_batches_1, remaining_batches_2


def process_paired_batches(paired_batches, dist_thresh=80):
    final_dict = {}
    for paired_batch in paired_batches:
        set_1, set_2 = paired_batch[0], paired_batch[1]
        paired_strings, removed_strings, added_strings = [], [], []
        cord1, string_sets_1, pos_sets_1, alt_set_1 = (
            set_1[1],
            set_1[-4],
            set_1[-2],
            set_1[-1],
        )
        cord2, string_sets_2, pos_sets_2, alt_set_2 = (
            set_2[1],
            set_2[-4],
            set_2[-2],
            set_2[-1],
        )
        n_list, idx_list = [], []
        if len(string_sets_1) > 1:
            string_sets_1 = [item for sublist in string_sets_1 for item in sublist]
            pos_sets_1 = [
                (item[0] + (n * 256), item[1] + (n * 256))
                for n, sublist in enumerate(pos_sets_1)
                for item in sublist
            ]
        if len(string_sets_2) > 1:
            string_sets_2 = [item for sublist in string_sets_2 for item in sublist]
            pos_sets_2 = [
                (item[0] + (n * 256), item[1] + (n * 256))
                for n, sublist in enumerate(pos_sets_2)
                for item in sublist
            ]
        comb_array_1 = np.array([string_sets_1, pos_sets_1])
        comb_array_2 = np.array([string_sets_2, pos_sets_2])
        if len(comb_array_1.shape) > 2:
            comb_array_1 = comb_array_1[:, 0, :]
        if len(comb_array_2.shape) > 2:
            comb_array_2 = comb_array_2[:, 0, :]
        for n, pos_1 in enumerate(comb_array_1[1]):
            dists = [
                np.linalg.norm((pos_1[0] - pos_2[0], pos_1[1] - pos_2[1]))
                for pos_2 in comb_array_2[1]
            ]
            if np.min(dists) < dist_thresh:
                idx = np.argmin(dists)
                if type(string_sets_1[0]) is not list:
                    string_sets_1 = [string_sets_1]
                if type(string_sets_2[0]) is not list:
                    string_sets_2 = [string_sets_2]
                paired_strings.append((string_sets_1[0][n], string_sets_2[0][int(idx)]))
                n_list.append(n)
                idx_list.append(idx)
            removed_strings = [
                batch for n, batch in enumerate(string_sets_1) if n not in n_list
            ]
            added_strings = [
                batch for n, batch in enumerate(string_sets_2) if n not in idx_list
            ]
            tmp = tuple(list(cord1) + [float(alt_set_1)])
            # final_dict[cord1 + [alt_set_1]] = paired_strings, removed_strings, added_strings
            final_dict[tmp] = paired_strings, removed_strings, added_strings
    return final_dict


def get_changes(run1_npy_file, run2_npy_file):
    run1_info = np.load(run1_npy_file, allow_pickle=True)
    run2_info = np.load(run2_npy_file, allow_pickle=True)
    results_1 = get_detection_sets(run1_info)
    results_2 = get_detection_sets(run2_info)
    if len(results_1) == 0 or len(results_2) == 0:
        paired_sets, text_set_removed, text_set_added = [], [], []
        coords = []
        paired = {}
    # for result_1 in results_1[3]:
    #     print(result_1[-3])
    else:
        processed_results_1 = [pick_high_conf(result)[1] for result in results_1]
        processed_results_2 = [pick_high_conf(result)[1] for result in results_2]
        paired_sets, text_set_removed, text_set_added = align_processed_batches(
            processed_results_1, processed_results_2
        )
        paired = process_paired_batches(paired_sets)
        coords = list(paired.keys())
    change_list = []
    major_category = "Gantry"
    minor_category = "Text"
    for idn, coord in enumerate(coords):
        obj_chg_status = []
        strings_pairs, strings_added, strings_removed = paired[coord]
        modified_pairs = [
            string_pair
            for string_pair in strings_pairs
            if textdistance.ratcliff_obershelp(string_pair[0], string_pair[1]) < 0.5
        ]
        # if len(strings_added) + len(strings_removed) + len(modified_pairs) > 0:
        #     desc = "change_detected"
        if len(strings_added) + len(strings_removed) > 0:
            desc = "change_detected"
        else:
            # desc = ""
            continue
        if len(strings_added) > 0:
            obj_chg_status = ["Added"]
            tmp = str(",".join([str(c) for c in coord]))
            change_list.append(
                [
                    "".join(obj_chg_status),
                    major_category,
                    minor_category,
                    "",
                    desc,
                    "",
                    1,
                    "",
                    "",
                    "",
                    "",
                    [tmp],
                    [tmp],
                    "CRDv2",
                    "",
                    "",
                    json.dumps({"Added": strings_added}),
                ]
            )
        if len(strings_removed) > 0:
            obj_chg_status = ["Removed"]
            tmp = str(",".join([str(c) for c in coord]))
            change_list.append(
                [
                    "".join(obj_chg_status),
                    major_category,
                    minor_category,
                    "",
                    desc,
                    "",
                    1,
                    "",
                    "",
                    "",
                    "",
                    [tmp],
                    [tmp],
                    "CRDv2",
                    "",
                    "",
                    json.dumps({"Removed": strings_removed}),
                ]
            )
        # if len(modified_pairs) > 0:
        #     obj_chg_status += ["text modified "]
        # change_list.append(["".join(obj_chg_status), major_category, minor_category, "", desc, "", 1, "", "", "", "",
        #                     coord, coord, "CRDv2", "", "",
        #                     json.dumps(
        #                         {"modified": modified_pairs, "added": strings_added, "removed": strings_removed})])
        # tmp = str(",".join([str(c) for c in coord]))
        # change_list.append(["".join(obj_chg_status), major_category, minor_category, "", desc, "", 1, "", "", "", "",
        #                     [tmp], [tmp], "CRDv2", "", "",
        #                     json.dumps(
        #                         {"Added": strings_added, "Removed": strings_removed})])
    return change_list


def get_text_crd(
    engine,
    run1="/data/projects/crd_project/test_dataset/comparison_augsburg/"
    "_data_aiim_change_detection_comparison07-augsburg_/20190827/2019-08-271252_GPS_Photo.txt",
    run2="/data/projects/crd_project/test_dataset/comparison_augsburg/"
    "_data_aiim_change_detection_comparison07-augsburg_/20190703/2019-07-031334_GPS_Photo.txt",
    cache=".",
):
    run1_npy = "{}/text_detections/{}.npy".format(cache, os.path.basename(run1))
    run2_npy = "{}/text_detections/{}.npy".format(cache, os.path.basename(run2))
    cache_dir = "{}/text_detections/".format(cache)
    if not os.path.exists(run1_npy):
        write_dets_npy(
            engine, run1, npy_name=os.path.basename(run1), save_dir=cache_dir
        )
    if not os.path.exists(run2_npy):
        write_dets_npy(
            engine, run2, npy_name=os.path.basename(run2), save_dir=cache_dir
        )
    return get_changes(run1_npy, run2_npy)


if __name__ == "__main__":
    import csv
    from segmentation import Segmenter

    engine = Segmenter("./cache")
    changes = get_text_crd(engine)

    with open("./output4.csv", mode="w") as changes_file:
        csv_writer = csv.writer(
            changes_file, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )

        csv_writer.writerow(
            [
                "ID",
                "Object Change Status",
                "Major Category",
                "Minor Category",
                "Description",
                "Cause of change",
                "Created  time",
                "Amount",
                "Length",
                "construction type",
                "Start Or End?",
                "Construction detail",
                "GPS Coordinate1",
                "GPS Coordinate2",
                "Created By",
                "Road side",
                "Traffic sign type",
                "Detail",
            ]
        )

        for i, change in enumerate(changes, 1):
            csv_writer.writerow([i] + change)
