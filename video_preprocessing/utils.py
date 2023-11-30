# Copyright (C) 2020 by NavInfo Europe B.V. The Netherlands - All rights reserved
# Information classification: Confidential
# This content is protected by international copyright laws.
# Reproduction and distribution is prohibited without written permission.
import os
import subprocess
import logging
import pandas as pd
from shutil import move


def _parse_dirname_img(video_path, level):
    def tokenize_video_path(video_path):
        """
        This function tokenizes the video path
        :return: year month, day, hour, minutes
        """
        tokens = video_path.split("_")
        date_str = tokens[0]
        time_str = tokens[1]

        year = int(date_str[0:4])
        month = int(date_str[4:6])
        day = int(date_str[6:8])

        hour = int(time_str[0:2])
        minute = int(time_str[2:4])
        return year, month, day, hour, minute

    year, month, day, hour, minute = tokenize_video_path(video_path=video_path)

    if level == 1:
        return "{0:04d}{1:02d}{2:02d}".format(year, month, day)
    elif level == 2:
        return "{0:04d}-{1:02d}-{2:02d}{3:02d}{4:02d}_GPS_Photo".format(
            year, month, day, hour, minute
        )
    elif level == 3:
        return "{0:02}/{1:02}/{2:02}".format(month, day, hour)


def get_filepath_gps_from_mp4(src_video_path, outdir):
    filename = os.path.basename(src_video_path)
    filename, ext = os.path.splitext(filename)
    assert ext == ".mp4"
    gps_run_filename = os.path.join(outdir, filename + "_exiftool_gps.csv")
    nmea_run_filename = os.path.join(outdir, filename + "_exiftool_nmea.csv")
    return gps_run_filename, nmea_run_filename


def extract_gps_info(src_video_file, out_gps_file):
    command = (
        "exiftool -ee "
        "-coordFormat %02.8f "
        "-dateFormat %Y:%m:%d-%H:%M:%S "
        "-SampleTime -GPSDateTime -GPSLatitude "
        "-GPSLongitude -GPSTrack -GPSSpeed -StartTime "
        "{0} > {1}".format(src_video_file, out_gps_file)
    )
    subprocess.call(command, shell=True)


def extract_nmea_info(src_video_file, out_nmea_file):
    command = "exiftool -ee -a -GPSLog -b " "{0} > {1}".format(
        src_video_file, out_nmea_file
    )
    subprocess.call(command, shell=True)


# def extract_metadata_info(src_video_file, out_metadata_file):
#     logging.info(msg="Metadata info not yet implemented")


def concat_csv_nmea_files(drive_dir, gps_photo_dir):
    csv_filelist = [
        os.path.join(root, filename)
        for root, subdirs, files in os.walk(drive_dir)
        for filename in files
        if filename.endswith("_gps_nmea.csv")
    ]

    dfs = list()
    for filename in csv_filelist:
        dfs.append(pd.read_csv(filename))
    df_combined = pd.concat(dfs)
    df_combined = df_combined.sort_values(by="FILE_URL", ascending=True)
    df_combined.set_index("GUID")

    combined_filename = os.path.normpath(gps_photo_dir).split(os.sep)[-1] + ".txt"
    df_combined.to_csv(os.path.join(drive_dir, combined_filename), index=False)
    [os.remove(filename) for filename in csv_filelist]


def cleanup(output_dir, run_output_dir):
    logging.info("Final cleanup of redundant files")
    # recursive cleanup of the output dir of redundant image files
    [
        os.remove(os.path.join(root, filename))
        for root, subdirs, files in os.walk(output_dir)
        for filename in files
        if filename.startswith("frame_")
    ]

    # recursive cleanup of temp GPS files
    [
        os.remove(os.path.join(root, filename))
        for root, subdirs, files in os.walk(output_dir)
        for filename in files
        if filename.endswith("exiftool_gps.csv")
        or filename.endswith("exiftool_nmea.csv")
    ]

    # move drive CSV files to output folder
    [
        move(os.path.join(root, filename), os.path.join(output_dir, filename))
        for root, subdirs, files in os.walk(run_output_dir)
        for filename in files
        if filename.endswith("_gps_nmea.csv")
    ]

    # cleanup - empty directory
    for root, dirs, files in os.walk(output_dir, topdown=False):
        for name in dirs:
            # check whether the directory is empty
            if len(os.listdir(os.path.join(root, name))) == 0:
                logging.info("Deleting {0}".format(os.path.join(root, name)))
                try:
                    os.rmdir(os.path.join(root, name))
                except:
                    logging.error(
                        "Failed to delete: {0}".format(os.path.join(root, name))
                    )
