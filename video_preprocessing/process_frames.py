# Copyright (C) 2020 by NavInfo Europe B.V. The Netherlands - All rights reserved
# Information classification: Confidential
# This content is protected by international copyright laws.
# Reproduction and distribution is prohibited without written permission.
# Author: Ratnajit Mukherjee, May 2020
import os
import logging
import numpy as np
from shutil import move


def filepath_truncated(filepath):
    filepath_arr = os.path.normpath(filepath).split(os.sep)
    index = [i for i, token in enumerate(filepath_arr) if "_Photo" in token][0]

    init_str = filepath_arr[index].split("_")[-1]
    substr = "/".join(filepath_arr[index + 1 : len(filepath_arr)])
    truncated_path = os.path.join(init_str, substr)
    truncated_path = truncated_path.replace("/", "\\")
    return truncated_path


def process_frame(
    df_interp, output_dir, drive_dir, index, frame_sample_time, frame_format
):
    """
    Renames a frame to the final output format
    """
    if frame_sample_time not in df_interp.index:
        return

    gps_time_stamp_arr = str(df_interp.loc[frame_sample_time, "GPSDateTime"]).split(" ")
    gps_time_only = gps_time_stamp_arr[1].split(":")
    second_arr = gps_time_only[-1].split(".")
    if len(second_arr) > 1:
        second, microsec = second_arr
    else:
        second = second_arr[0]
        microsec = "000000"
    time_stamp_final = (
        gps_time_stamp_arr[0]
        + gps_time_only[0]
        + gps_time_only[1]
        + second
        + microsec[:3]
    )
    lat = df_interp.loc[frame_sample_time, "GPSLatitude"]
    lon = df_interp.loc[frame_sample_time, "GPSLongitude"]

    # The interpolation doesn't extrapolate values outside of original data
    if np.isnan(lat) or np.isnan(lon):
        return

    # obtaining the old frame name
    old_filepath = os.path.join(output_dir, frame_format % index)

    # new file name
    new_filename = "{0}.jpg".format(time_stamp_final)
    new_filepath = os.path.join(output_dir, new_filename)
    if not os.path.isfile(old_filepath):
        # check for faulty videos (which stop in the middle)
        return
    else:
        move(old_filepath, new_filepath)

    """
    NOTE: for now we keep the absolute path because it might break everyone's code
    Before we ship this, we change this to the truncated path as per requirements
    """
    # truncated_path = filepath_truncated(new_filepath)

    df_interp.loc[frame_sample_time, "Filepath"] = new_filepath
