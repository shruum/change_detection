# import sys
# from IPython.core import ultratb
# sys.excepthook = ultratb.FormattedTB(mode="Verbose", color_scheme="Linux", call_pdb=1)

import logging
import os
import sys
from typing import Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import imread
from lanes.detector import LaneDetector
from lanes.utils import (
    apply_custom_color_map,
    get_angle,
    intersections,
    my_color_rgb,
    my_white_rgb,
    remove_lines_of_similar_slope,
    remove_outliers,
    remove_points_far_from_center,
)


from lanes.vpoint_debug import vpoint_vis1, vpoint_vis1b, vpoint_vis2, vpoint_vis3


def localize_vanishing_point_one(img_path, segmenter, debug=False) -> Optional[Tuple]:
    """Returns mean and std point if succesfull.
       Returns None on fail."""
    img = imread(img_path)
    seg = segmenter.segment(img_path)

    seg3 = cv2.cvtColor(seg, cv2.COLOR_GRAY2RGB)
    seg3w = apply_custom_color_map(seg3, [2, 1], my_white_rgb)
    seg1w = cv2.cvtColor(seg3w, cv2.COLOR_RGB2GRAY)

    vpoint_prior = (img.shape[1] // 2, img.shape[0] // 2)
    detector = LaneDetector(vpoint_prior)

    can = cv2.Canny(img, detector.canny_th1, detector.canny_th2)

    dil_size = 3
    element = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (dil_size * 2, dil_size * 2), (dil_size, dil_size)
    )
    seg1wf = cv2.dilate(seg1w, element)

    can_x_seg1wf = cv2.bitwise_and(can, seg1wf)

    lines = detector._detect_lines(can_x_seg1wf)
    unique_lines = remove_lines_of_similar_slope(lines)
    points = intersections(unique_lines)
    center_points = remove_points_far_from_center(points, img.shape)

    if debug:
        vpoint_vis1(img, seg, seg3w, seg3, can, seg1wf, can_x_seg1wf)
        vpoint_vis1b(img, lines, unique_lines, points, center_points)

    if len(center_points) == 0:
        return vpoint_prior

    # results per image
    point, point_std = np.mean(center_points, 0), np.std(center_points, 0)

    if debug:
        vpoint_vis2(img, center_points, point, point_std)

    if point_std[0] < 5 and point_std[1] < 5:
        return point
    else:
        return vpoint_prior


def localize_vanishing_point(img_paths, segmenter, debug=False):
    vpoints = [
        point
        for img_path in tqdm(img_paths)
        if (point := localize_vanishing_point_one(img_path, segmenter, debug))
        is not None
    ]

    if len(vpoints) < 1:
        raise Exception(
            """Vaninshing Point Detection failed.
                Less than one image allowed for vpoint localization.
                Try feeding more images."""
        )

    vpoint, vpoint_std = np.mean(vpoints, 0), np.std(vpoints, 0)

    if debug:
        vpoint_vis3(img_paths[-1], vpoint, vpoint_std)

    vpoint = int(vpoint[0]), int(vpoint[1])
    logging.info(f"vpoint: {vpoint}")
    return vpoint


def localize_vanishing_points(
    run1_paths, run2_paths, segmenter, cache_dir, debug=False
):
    data_file = os.path.join(cache_dir, "vpoint1.npy")
    if not os.path.exists(data_file):
        vpoint1 = localize_vanishing_point(run1_paths, segmenter, debug)
        logging.info(f"Caching {data_file}")
        np.save(data_file, vpoint1)
    else:
        logging.info(f"Reading from cached {data_file}")
        vpoint1 = np.load(data_file)

    data_file = os.path.join(cache_dir, "vpoint2.npy")
    if not os.path.exists(data_file):
        vpoint2 = localize_vanishing_point(run2_paths, segmenter, debug)
        logging.info(f"Caching {data_file}")
        np.save(data_file, vpoint2)
    else:
        logging.info(f"Reading from cached {data_file}")
        vpoint2 = np.load(data_file)

    return vpoint1, vpoint2
