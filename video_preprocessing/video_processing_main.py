# Copyright (C) 2020 by NavInfo Europe B.V. The Netherlands - All rights reserved
# Information classification: Confidential
# This content is protected by international copyright laws.
# Reproduction and distribution is prohibited without written permission.
"""
**************************************************************************************
COMMENTS FOR UNDERSTANDING:
---------------------------------
Blackvue camera has 3 outputs:
1) the video frames - obviously
2) the embedded GPS information - starttime, endtime, lat, lon, speed etc.
3) the GPS log - which contains metadata about the GPS information
NOTE: we are unable to extract altitude for now because exiftool DOES NOT support altitude yet.
Future versions will probably support this (but don't keep your hopes high).
**************************************************************************************
"""
import os
import ffmpeg
import logging
import numpy as np
import video_preprocessing.utils as utils
from video_preprocessing.blackvue_parser import BlackVueParser
from video_preprocessing.dataframe_operations import DataFrameOperations


class VideoProcessing:
    def __init__(self, input_dir, output_dir):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.FRAME_FORMAT = "frame_%06d.jpg"

    def extract_video_frames(self, src_video_path, output_dir_video):
        logging.info("Extracting frames from {0}".format(src_video_path))
        # probe into each video
        ffprobe_dict = ffmpeg.probe(src_video_path)
        video_stream = next(
            (
                stream
                for stream in ffprobe_dict["streams"]
                if stream["codec_type"] == "video"
            ),
            None,
        )
        num_frames = int(video_stream["nb_frames"])
        frame_rate = eval(video_stream["r_frame_rate"])

        # create output video directory if required
        if not os.path.exists(output_dir_video):
            os.makedirs(output_dir_video)

        # this is where you start selecting from the frames
        mod = 1
        vf_string = "select=not(mod(n\,%d))" % mod
        try:
            # Extracts frames to dir
            out, _ = (
                ffmpeg.input(src_video_path)
                .output(
                    os.path.join(output_dir_video, self.FRAME_FORMAT),
                    vf=vf_string,
                    vsync="vfr",
                    q=2,
                )
                .global_args("-loglevel", "quiet")
                .run()
            )
        except ffmpeg.Error as e:
            logging.error(e.stderr)
            logging.error(
                "Failed to extract frames from video: {0}".format(src_video_path)
            )
            return np.array([]), []

        frame_time = 1 / frame_rate
        if mod != 0:
            frame_time *= mod
        frame_times = [i * frame_time for i in range(0, num_frames)]

        return frame_times

    def extract_temp_gps_info_files(self, src_video_path, gps_file_run, nmea_file_run):
        logging.info("Extracting GPS and GPSLog information")
        # create the temp txt file to unload the GPS info.
        temp_gps_run = os.path.splitext(gps_file_run)[0] + ".txt"
        temp_nmea_run = os.path.splitext(nmea_file_run)[0] + ".txt"

        # now we run the GPS and NMEA extraction subprocess for the run:
        utils.extract_gps_info(src_video_file=src_video_path, out_gps_file=temp_gps_run)
        utils.extract_nmea_info(
            src_video_file=src_video_path, out_nmea_file=temp_nmea_run
        )
        return temp_gps_run, temp_nmea_run

    def extract_video_main(self):
        """
        This is the main function which controls the entire extraction of videos to images
        including the extraction of GPS and NMEA details, frame interpolation and final
        output to directories.
        :return:
        """
        filelist = sorted(
            [
                os.path.join(root, filename)
                for root, subdirs, files in os.walk(self.input_dir)
                for filename in files
                if filename.endswith(".mp4")
            ]
        )

        # this will be the name of the final folder with images from multiple sequences in a single drive
        _drive_dir = utils._parse_dirname_img(os.path.basename(filelist[0]), level=1)
        _drive_dir = os.path.join(self.output_dir, _drive_dir)
        os.makedirs(_drive_dir, exist_ok=True)

        # this is where the images will be extracted (within sub-folders)
        gps_photo_dir = utils._parse_dirname_img(os.path.basename(filelist[0]), level=2)
        gps_photo_dir = os.path.join(_drive_dir, gps_photo_dir)
        os.makedirs(gps_photo_dir, exist_ok=True)

        check_final_gps_txt = os.path.join(
            _drive_dir, os.path.normpath(gps_photo_dir).split(os.sep)[-1] + ".txt"
        )
        if os.path.isfile(check_final_gps_txt):
            logging.info(
                msg="Video from drive already extracted. Skipping video extraction"
            )
            return _drive_dir
        else:
            for video_path in filelist:
                drive_name = utils._parse_dirname_img(
                    video_path=os.path.basename(video_path), level=3
                )
                run_output_dir = os.path.join(gps_photo_dir, drive_name)
                os.makedirs(run_output_dir, exist_ok=True)

                csv_gps_run, csv_nmea_run = utils.get_filepath_gps_from_mp4(
                    video_path, run_output_dir
                )

                temp_gps_run, temp_nmea_run = self.extract_temp_gps_info_files(
                    src_video_path=video_path,
                    gps_file_run=csv_gps_run,
                    nmea_file_run=csv_nmea_run,
                )

                # putting the files in the list for cleaner argument passing
                gps_file_list = [temp_gps_run, csv_gps_run, temp_nmea_run, csv_nmea_run]

                # now we parse the 2 temp run files into CSV files
                blackvueparser = BlackVueParser()
                blackvueparser.black_vue_parser_main(gps_file_list=gps_file_list)

                frame_times_run = self.extract_video_frames(
                    src_video_path=video_path, output_dir_video=run_output_dir
                )

                # now we process the dataframes for each run
                data_frame_ops = DataFrameOperations()
                data_frame_ops.process_dataframes(
                    src_video=video_path,
                    output_dir=run_output_dir,
                    drive_dir=_drive_dir,
                    csv_gps=csv_gps_run,
                    csv_nmea=csv_nmea_run,
                    frame_times=frame_times_run,
                    frame_format=self.FRAME_FORMAT,
                )

                utils.cleanup(_drive_dir, run_output_dir)
            utils.concat_csv_nmea_files(
                drive_dir=_drive_dir, gps_photo_dir=gps_photo_dir
            )
            logging.info("Video extraction completed.")
            return _drive_dir


"""
*********************************************************************************
Use the main function for testing purpose. Else this file is called from main.py
*********************************************************************************
"""
if __name__ == "__main__":
    input_dir = "/data/aiim/change_detection/comparison07-augsburg/w35/videos"
    output_dir = "/volumes2/CRDoutputdir/augsburg"
    vid_procesing = VideoProcessing(input_dir=input_dir, output_dir=output_dir)
    vid_procesing.extract_video_main()
