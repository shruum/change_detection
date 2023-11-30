# -*- coding: utf-8 -*-

from tracker import DEFAULT_PARAMS


def get_arguments():

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_path",
        default="",
        type=str,
        help="Path to data (MP4 video or folder of images)",
    )
    parser.add_argument(
        "detection_path", type=str, help="Input directory for detection files"
    )
    parser.add_argument(
        "output_path",
        type=str,
        help="Absolute path to where the output files will be saved",
    )
    parser.add_argument(
        "--input_type",
        type=str,
        default=DEFAULT_PARAMS["input_type"],
        choices=["video", "image"],
        help="Type of input data to load",
    )
    parser.add_argument(
        "--output_filename",
        default=DEFAULT_PARAMS["output_filename"],
        type=str,
        help="Name of output file (excluding extension)",
    )
    parser.add_argument(
        "--method", type=str, default=DEFAULT_PARAMS["method"], choices=["ATOM", "DiMP"]
    )
    parser.add_argument(
        "--use_detection",
        action="store_true",
        help="Use detection for bounding box position, rather than tracking output. Typically use this when detector was trained on same data distribution.",
    )
    parser.add_argument(
        "--num_future_frames",
        default=DEFAULT_PARAMS["num_future_frames"],
        type=int,
        help="Number of frames to look ahead",
    )
    parser.add_argument(
        "--end_threshold",
        default=DEFAULT_PARAMS["end_threshold"],
        type=int,
        help="Number of frames in which object is not found to end track",
    )
    parser.add_argument(
        "--min_length",
        default=DEFAULT_PARAMS["min_length"],
        type=int,
        help="Minimum length of track for saving",
    )
    parser.add_argument(
        "--max_length",
        default=DEFAULT_PARAMS["max_length"],
        type=int,
        help="Maximum length of track",
    )
    parser.add_argument(
        "--upper_conf_threshold",
        default=DEFAULT_PARAMS["upper_conf_threshold"],
        type=float,
        help="Minimum confidence for bounding box initialization",
    )
    parser.add_argument(
        "--lower_conf_threshold",
        default=DEFAULT_PARAMS["lower_conf_threshold"],
        type=float,
        help="Minimum confidence for bounding box association",
    )
    parser.add_argument(
        "--iou_threshold",
        default=DEFAULT_PARAMS["iou_threshold"],
        type=float,
        help="Minimum IoU overlap for bounding box association",
    )
    parser.add_argument(
        "--min_bbox_area",
        default=DEFAULT_PARAMS["min_bbox_area"],
        type=float,
        help="Minimum area of bounding box",
    )
    parser.add_argument(
        "--img_height",
        default=DEFAULT_PARAMS["img_height"],
        type=int,
        help="Height of image",
    )
    parser.add_argument(
        "--img_width",
        default=DEFAULT_PARAMS["img_width"],
        type=int,
        help="Width of image",
    )
    parser.add_argument(
        "--max_relative_diff",
        default=DEFAULT_PARAMS["max_relative_diff"],
        type=float,
        help="Maximum relative difference between actual displacement and predicted displacement (default: not used)",
    )
    parser.add_argument(
        "--crop_window",
        default=DEFAULT_PARAMS["crop_window"],
        type=str,
        help="Top, bottom, left, right",
    )
    parser.add_argument(
        "--min_displacement",
        default=DEFAULT_PARAMS["min_displacement"],
        type=float,
        help="Minimum displacement of bounding box in {end_threshold} frames",
    )
    parser.add_argument(
        "--save_crop_images",
        action="store_true",
        help="Save all the cropped images of each class onto disk",
    )
    parser.add_argument(
        "--run_classifier",
        action="store_true",
        help="Runs a classifier on the tracks to get extra classes",
    )
    parser.add_argument(
        "--classifier_path", type=str, help="Path to saved classification model"
    )
    parser.add_argument(
        "--classifier_datatype",
        default=DEFAULT_PARAMS["classifier_datatype"],
        type=str,
        help="Load classifier data type to find classes",
    )
    parser.add_argument(
        "--classifier_datadir",
        default=DEFAULT_PARAMS["classifier_datadir"],
        type=str,
        help="Root path to classifier dataset",
    )
    parser.add_argument(
        "--classifier_image_size",
        default=DEFAULT_PARAMS["classifier_image_size"],
        type=int,
        help="Size of input image to classifier",
    )
    parser.add_argument(
        "--classification_window",
        default=DEFAULT_PARAMS["classification_window"],
        type=int,
        help="Size of window around initial detection to determine which crops to classify",
    )
    parser.add_argument(
        "--overlay_tracks",
        action="store_true",
        help="Overlay both detection and Tracks on images",
    )
    parser.add_argument(
        "--save_track_json",
        action="store_true",
        help="stores tracking information for each frame in json format",
    )

    return parser.parse_args()


if __name__ == "__main__":
    get_arguments()
