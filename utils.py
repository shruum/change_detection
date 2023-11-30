# Copyright (C) 2020 by NavInfo Europe B.V. The Netherlands - All rights reserved
# Information classification: Confidential
# This content is protected by international copyright laws.
# Reproduction and distribution is prohibited without written permission.
import os
import csv
import cv2
import uuid
import logging
import numpy as np
import pandas as pd


def imread(filename):
    logging.debug("Reading image from file:", filename)
    img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    if len(img.shape) == 3:
        return img[:, :, ::-1]
    else:
        return img


def imwrite(filename, contents):
    contents = contents.astype("uint8")
    if len(contents.shape) > 2:
        contents = contents[:, :, ::-1]
    cv2.imwrite(filename, contents)
    logging.debug("Written image to file:", filename)


def write_to_csv(changes, cache_dir):
    def _gps2str(gps_arr):
        for i in range(len(gps_arr)):
            x = ",".join(map(str, gps_arr))
        return x

    csv_filepath = os.path.join(cache_dir, "final_changes.csv")

    with open(csv_filepath, mode="w") as changes_file:
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
                "filename_run1",
                "filename_run2",
                "frame_start",
                "frame_end",
            ]
        )

        for i, change in enumerate(changes, 1):
            change[11] = _gps2str(change[11])
            change[12] = _gps2str(change[12])
            csv_writer.writerow([i] + change)

    # calling the function for the new format
    # aligned_file_list = os.path.join(cache_dir, "aligned_files.txt")
    # if os.path.isfile(aligned_file_list):
    #     new_csv_format(
    #         csv_file=csv_filepath,
    #         aligned_file_list=aligned_file_list,
    #         output_directory=cache_dir,
    #     )


def new_csv_format(csv_file, aligned_file_list, output_directory):
    # reading old csv file
    df_old = pd.read_csv(csv_file)
    df_old.fillna("", inplace=True)

    # reading aligned filelist
    aligned_array = np.loadtxt(aligned_file_list, dtype=str, delimiter=",")
    logging.info(f"File loaded from {aligned_file_list}")

    aligned_filelist1 = list(aligned_array[:, 0])
    aligned_filelist2 = list(aligned_array[:, 1])

    column_names = [
        "ID",
        "Name",
        "Error categories",
        "Big classification",
        "Small classification",
        "Error Description",
        "CreateTime",
        "Photo_1",
        "Photo_2",
        "MapNumber",
        "Layer",
        "MainOID",
        "AcceptState",
        "WorkState",
        "Comment",
        "MarkPosition",
        "PhotoPosition_1",
        "PhotoPosition_2",
        "Location",
        "Cause",
        "CreateUser",
        "ModifyUser",
        "ModifyTime",
        "Guid",
        "LineWKT",
        "ConstructionID",
        "ConstructionLogo",
        "ConstructionType",
        "ConstructionLineLength",
        "ConstructionLineWKT",
    ]

    df_new = pd.DataFrame(columns=column_names)
    for index, row in df_old.iterrows():
        df_new.loc[index, "ID"] = row["ID"]
        df_new.loc[index, "Error categories"] = row["Object Change Status"]
        df_new.loc[index, "Big classification"] = "Object"
        df_new.loc[index, "Small classification"] = row["Minor Category"]
        df_new.loc[index, "Error Description"] = (
            row["Major Category"] + " " + row["Object Change Status"]
        )

        # match the filename and full path
        matched_filename_run1 = [
            filename
            for filename in aligned_filelist1
            if row["filename_run1"] in filename
        ][0]
        matched_filename_run2 = [
            filename
            for filename in aligned_filelist2
            if row["filename_run2"] in filename
        ][0]

        df_new.loc[index, "Photo_1"] = matched_filename_run1
        df_new.loc[index, "Photo_2"] = matched_filename_run2

        df_new.loc[index, "PhotoPosition_1"] = row["GPS Coordinate1"]
        df_new.loc[index, "PhotoPosition_2"] = row["GPS Coordinate2"]

        df_new.loc[index, "Guid"] = uuid.uuid4()

    df_new.fillna("", inplace=True)
    df_new.set_index("ID")
    df_new.to_csv(os.path.join(output_directory, "final_changes_new.csv"), index=False)


def get_mapping_lists(run1_path_list, run2_path_list, cache_dir):
    # Deal with duplicate frames
    def find_frame_mapping(inverse_indices_list):
        mapping_list = []
        for frame in np.unique(inverse_indices_list):
            mapped_frame = np.nonzero(inverse_indices_list == frame)[0]
            if len(mapped_frame) > 0:
                mapped_frame_start = mapped_frame[0]
                mapped_frame_end = mapped_frame[-1]
            else:  # Do identity mapping
                mapped_frame_start = mapped_frame
                mapped_frame_end = mapped_frame
            mapping_list.append(np.array([mapped_frame_start, mapped_frame_end]))
        return mapping_list

    mapping_list_run1_path = os.path.join(cache_dir, "mapping_list_run1.npy")
    mapping_list_run2_path = os.path.join(cache_dir, "mapping_list_run2.npy")
    if not os.path.isfile(mapping_list_run1_path) or not os.path.isfile(
        mapping_list_run2_path
    ):
        run1_path_list_unique, inverse_indices_run1 = np.unique(
            np.array(run1_path_list), return_inverse=True
        )
        run2_path_list_unique, inverse_indices_run2 = np.unique(
            np.array(run2_path_list), return_inverse=True
        )
        run1_path_list_unique = run1_path_list_unique.tolist()
        run2_path_list_unique = run2_path_list_unique.tolist()
        mapping_list_run1 = find_frame_mapping(inverse_indices_run1)
        mapping_list_run2 = find_frame_mapping(inverse_indices_run2)
        np.save(mapping_list_run1_path, [run1_path_list_unique, mapping_list_run1])
        np.save(mapping_list_run2_path, [run2_path_list_unique, mapping_list_run2])
    else:
        run1_path_list_unique, mapping_list_run1 = np.load(
            mapping_list_run1_path, allow_pickle=True
        )
        run2_path_list_unique, mapping_list_run2 = np.load(
            mapping_list_run2_path, allow_pickle=True
        )
    mapping_list_run1 = np.array(mapping_list_run1)
    mapping_list_run2 = np.array(mapping_list_run2)
    return (
        run1_path_list_unique,
        run2_path_list_unique,
        mapping_list_run1,
        mapping_list_run2,
    )


def get_change_per_frame(change_list, min_frame, max_frame):
    # change_list: [Nchanges X [add/remove, class_name, frame_start, frame_end]]

    # change per frame
    frame_array = []
    for frame_id in range(min_frame, max_frame + 1):
        frame_array.append(frame_id)
    frame_array = np.array(frame_array)
    change_per_frame = np.empty((max_frame - min_frame + 1, 4), dtype=object)
    change_per_frame[:, 0] = frame_array

    for change in change_list:
        change_type = str(change[0])
        change_major_label = str(change[1])
        change_minor_label = str(change[2])
        change_min_frame = int(change[3])
        change_last_frame = int(change[4])
        change_max_frame = change_last_frame + 1  # this is required for indexing
        for frame_id in range(change_min_frame, change_max_frame):
            index = np.nonzero(frame_array == frame_id)[0][0]
            if change_per_frame[index, 1] is None:
                for i in range(1, 4):
                    change_per_frame[index, i] = []
            change_per_frame[index, 1].append(change_type)
            change_per_frame[index, 2].append(change_major_label)
            change_per_frame[index, 3].append(change_minor_label)

    return change_per_frame


if __name__ == "__main__":
    csv_file = "/data/projects/crd_project/cache/_data_aiim_change_detection_ChangeDetection_sampledata_Augsburg_/final_changes.csv"
    aligned_file_list = "/data/projects/crd_project/cache/_data_aiim_change_detection_ChangeDetection_sampledata_Augsburg_/aligned_files.txt"
    output_dir = "/volumes2/CRDoutputdir"
    new_csv_format(
        csv_file=csv_file,
        aligned_file_list=aligned_file_list,
        output_directory=output_dir,
    )
