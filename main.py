# Copyright (C) 2020 by NavInfo Europe B.V. The Netherlands - All rights reserved
# Information classification: Confidential
# This content is protected by international copyright laws.
# Reproduction and distribution is prohibited without written permission.
import csv
import logging
import os
import glob
import numpy as np

from options import parse_args
from video_preprocessing.video_processing_main import VideoProcessing
from dataloader import Loader, write_images

from detection import Detector
from segmentation import Segmenter

from tracking import Tracker
from lanes.vpoint import localize_vanishing_points
from lanes.change_detector import detect_lane_changes
from traffic_signs.change_detector import detect_traffic_sign_changes
from overhead_structure_face_and_traffic_barrier.overhead_structure_face_change_detector import (
    detect_overhead_structure_changes,
)
from overhead_structure_face_and_traffic_barrier.traffic_barrier_change_detector import (
    detect_traffic_barrier_changes,
)

from geo_CRD import get_text_crd
from arrows.change_detector import detect_arrow_changes
from utils import write_to_csv
from visualization.visualize_change_detection import VisualizeChangeDetection


def set_logging_level(quiet: bool, verbose: bool):
    log_level = logging.INFO
    if args.quiet:
        log_level = logging.WARNING
    if args.verbose:
        log_level = logging.DEBUG
    root_logger = logging.getLogger()

    console_handler = logging.StreamHandler()
    log_formatter = logging.Formatter("[%(levelname)s] - %(message)s")
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)
    root_logger.setLevel(log_level)


def extract_videos_if_needed(args):
    if args.format == "blackvue_video":
        logging.info("Extracting images from videos...")
        assert os.path.isdir(args.run1), "run1 must be a directory path"
        assert os.path.isdir(args.run2), "run2 must be a directory path"
        assert os.path.isdir(args.cache_dir), "cache_dir must be a directory path"
        video_processing = VideoProcessing(args.run1, args.cache_dir)
        args.run1 = video_processing.extract_video_main()
        video_processing = VideoProcessing(args.run2, args.cache_dir)
        args.run2 = video_processing.extract_video_main()
    return args.run1, args.run2


def align_images_if_needed(args):
    logging.info("Aligning image pairs...")
    aligned_files_path = os.path.join(args.cache_dir, "aligned_files.txt")
    if not os.path.exists(aligned_files_path):
        if args.format == "blackvue_images":
            aligned_array = Loader(
                args.run1,
                args.run2,
                no_alignment_speed_correction=args.no_speed_correction,
                format="blackvue",
            ).result
            aligned_cords = None
        elif args.format == "blackvue_video" or args.format == "aiim":
            if args.format == "blackvue_video":
                txt_file_1, txt_file_2 = (
                    glob.glob(os.path.join(args.run1, "*.txt"))[0],
                    glob.glob(os.path.join(args.run2, "*.txt"))[0],
                )
            elif args.format == "aiim":
                txt_file_1 = args.run1
                txt_file_2 = args.run2
            aligned_loader = Loader(
                txt_info_1=txt_file_1,
                txt_info_2=txt_file_2,
                no_alignment_speed_correction=args.no_speed_correction,
                format="aiim",
            )
            aligned_array = aligned_loader.result
            aligned_cords = aligned_loader.result_cords
        elif args.format == "fangzhou":
            aligned_loader = Loader(
                txt_info_1=args.run1,
                txt_info_2=args.run2,
                no_alignment_speed_correction=args.no_speed_correction,
                format=args.format,
            )
            aligned_array = aligned_loader.result
            aligned_cords = aligned_loader.result_cords
        else:
            print("No supported format selected")
            aligned_array = []
            aligned_cords = []
        if args.write_images:
            write_images(aligned_array, aligned_cords, args.cache_dir)
        aligned_array = np.reshape(
            aligned_array, newshape=(-1, 2)
        )  # Just to avoid errors later
        flat_coords = np.reshape(aligned_cords, newshape=(-1, 6))
        if flat_coords.shape[0] > 0:
            np.savetxt(
                aligned_files_path,
                np.concatenate((aligned_array, flat_coords), axis=-1),
                fmt="%s,%s,%s,%s,%s,%s,%s,%s",
            )
            logging.info(f"File saved at {aligned_files_path}")
        flat_coords = flat_coords.astype(np.float)
    else:
        aligned_array = np.loadtxt(aligned_files_path, dtype=str, delimiter=",")[:, :2]
        flat_coords = np.loadtxt(aligned_files_path, dtype=str, delimiter=",")[
            :, 2:
        ].astype(np.float)
        logging.info(f"File loaded from {aligned_files_path}")

    return (
        list(aligned_array[:, 0]),
        list(aligned_array[:, 1]),
        list(flat_coords[:, :3]),
        list(flat_coords[:, 3:]),
    )


def get_models(args, classification_path=None):
    return Detector(), Segmenter(args.cache_dir), Tracker(classification_path)


def get_changes(
    run1_paths, run2_paths, run1_coords, run2_coords, detector, segmenter, tracker, args
):
    logging.info("Vanishing Point Localization...")
    vpoints = localize_vanishing_points(
        run1_paths[50:150], run2_paths[50:150], segmenter, args.cache_dir
    )
    changes = list()
    logging.info("Detecting Lane Change...")
    # changes += detect_lane_changes(run1_paths, run2_paths, segmenter, vpoints, args.cache_dir)
    logging.info("Detecting Text Change...")
    # changes += get_text_crd(segmenter, txt_file1, txt_file2, cache=args.cache_dir)
    logging.info("Detecting Arrow Change...")
    changes += detect_arrow_changes(
        run1_paths, run2_paths, run1_coords, run2_coords, segmenter,
    )
    logging.info("Detecting Traffic Sign Change...")
    changes += detect_traffic_sign_changes(
        detector=detector,
        run1_coords=run1_coords,
        run2_coords=run2_coords,
        run1_path_list=run1_paths,
        run2_path_list=run2_paths,
        tracker=tracker,
        cache_dir=args.cache_dir,
        vanishing_points_run1=vpoints[0],
        vanishing_points_run2=vpoints[1],
    )
    # logging.info("Detecting OSF Change...")
    # changes += detect_overhead_structure_changes(
    #     run1_paths, run2_paths, run1_coords, run2_coords, segmenter, vpoints=vpoints
    # )
    # logging.info("Detecting TB Change...")
    # changes += detect_traffic_barrier_changes(
    #     run1_paths, run2_paths, run1_coords, run2_coords, segmenter, vpoints=vpoints
    # )
    return changes


def run_fangzhou_demo(
    args,
    trace_pair_path="/data/nie/teams/arl/datasets/CRD_Fangzhou/image/pair_of_traces.csv",
):

    data_dir = "/data/nie/teams/arl/datasets/CRD_Fangzhou/image/image series/"
    classification_path = "/data/nie/teams/arl/models_checkpoints/ark/quality_check_tool/1.1/classifier.pth"
    MIN_IMAGES = 100
    VISUALIZE = True

    # trace_pair_path: path to `pairs of traces.xls`
    trace_pairs = []
    with open(trace_pair_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        for row in csv_reader:
            if "order1" not in row:
                trace_pairs.append(row)
    trace_pairs = np.array(trace_pairs)

    cache_dir = args.cache_dir

    for trace_pair in trace_pairs:
        trace_pair_name = trace_pair[0] + "_" + trace_pair[1]
        logging.info("Processing trace pair:", trace_pair_name)
        args.cache_dir = os.path.join(cache_dir, trace_pair_name)
        os.makedirs(args.cache_dir, exist_ok=True)
        args.run1 = os.path.join(data_dir, trace_pair[0], "data", "trace.txt")
        args.run2 = os.path.join(data_dir, trace_pair[1], "data", "trace.txt")
        run1_paths, run2_paths, run1_coords, run2_coords = align_images_if_needed(args)
        if not os.path.isfile(os.path.join(args.cache_dir, "final_changes.csv")):
            detector, segmenter, tracker = get_models(
                args, classification_path
            )  # TODO: do this only use, remove args.cache_dir in seg init

            if len(run1_paths) > MIN_IMAGES and len(run2_paths) > MIN_IMAGES:
                changes = get_changes(
                    run1_paths,
                    run2_paths,
                    run1_coords,
                    run2_coords,
                    detector,
                    segmenter,
                    tracker,
                    args,
                )
            else:
                changes = []
            write_to_csv(changes=changes, cache_dir=args.cache_dir)
        if VISUALIZE:
            visualizer = VisualizeChangeDetection(
                run1_paths, run2_paths, args.cache_dir
            )
            visualizer.visualize_runs()


if __name__ == "__main__":

    args = parse_args()
    run_fangzhou_demo(args)

    #
    # set_logging_level(args.quiet, args.verbose)
    # rn1, rn2 = extract_videos_if_needed(args)
    # txt_file1, txt_file2 = (
    #     glob.glob(os.path.join(rn1, "*txt"))[0],
    #     glob.glob(os.path.join(rn2, "*txt"))[0],
    # )
    #
    # detector, segmenter, tracker = get_models(args)
    #
    # run1_paths, run2_paths, run1_coords, run2_coords = align_images_if_needed(args)
    # changes = get_changes(run1_paths, run2_paths, run1_coords, run2_coords, detector, segmenter, tracker, args)
    #
    # # CONCLUSION
    # write_to_csv(changes=changes, cache_dir=args.cache_dir)
