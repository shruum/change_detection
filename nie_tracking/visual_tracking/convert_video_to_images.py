#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import glob
import os

from options import get_arguments


def detect_in_videos(parent_folder, output_path):
    video_paths = glob.glob(os.path.join(parent_folder, "**", "*.mp4"), recursive=True)
    for video_path in video_paths:
        basename = os.path.splitext(os.path.basename(video_path))[0]
        output_folder = os.path.join(ARGS.output_path, basename)
        if not os.path.isdir(output_folder):
            print("Reading video:", video_path)
            os.makedirs(output_folder)
            cap = cv2.VideoCapture(video_path)
            frame_id = 0
            ret = True
            while cap.isOpened() and ret:
                frame_id_str = str(frame_id).zfill(6)
                print("Processing frame: ", frame_id_str)
                ret, image = cap.read()
                if ret:
                    output_filename = os.path.join(output_folder, frame_id_str + ".png")
                    cv2.imwrite(output_filename, image)
                    frame_id += 1


if __name__ == "__main__":
    ARGS = get_arguments()
    detect_in_videos(ARGS.input_path, ARGS.output_path)
