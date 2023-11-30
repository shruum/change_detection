import math
from itertools import combinations
from typing import Optional, Tuple

import numpy as np
import cv2


def my_color_name(cls_id):
    name_dict = {
        0: "black",
        1: "red",
        2: "green",
        5: "blue",
        7: "yellow",
        9: "magenta",
        99: "white",
    }
    return name_dict.get(cls_id, "black")


def my_color_rgb(cls_id):
    rgb_dict = {
        1: [255, 0, 0],
        2: [0, 255, 0],
        5: [0, 0, 255],
        7: [0, 255, 255],
        9: [255, 0, 255],
        99: [255, 255, 255],
    }
    return rgb_dict.get(cls_id, [0, 0, 0])


def my_white_rgb(cls_id):
    rgb_dict = {
        1: [255, 255, 255],
        2: [255, 255, 255],
    }
    return rgb_dict.get(cls_id, [0, 0, 0])


def apply_custom_color_map(im_gray, class_ids, my_fun):
    lut = np.zeros((256, 1, 3), dtype=np.uint8)
    for i in class_ids:
        lut[i, :, :] = my_fun(i)
    im_color = cv2.LUT(im_gray, lut)
    return im_color


def get_angle(line):
    return math.atan2(line[3] - line[1], line[2] - line[0]) / np.pi * 180


def line_intersection(line, line2) -> Optional[Tuple]:
    """Returns intersection point between two given lines.
       For parallel lines returns None.
    """
    x1, y1, x2, y2 = line
    x3, y3, x4, y4 = line2
    dx12 = x1 - x2
    dx34 = x3 - x4
    dy12 = y1 - y2
    dy34 = y3 - y4
    denom = dx12 * dy34 - dy12 * dx34
    try:
        px = (float(x1 * y2 - y1 * x2) * dx34 - dx12 * float(x3 * y4 - y3 * x4)) / denom
        py = (float(x1 * y2 - y1 * x2) * dy34 - dy12 * float(x3 * y4 - y3 * x4)) / denom
        return px, py
    except:
        return None


def intersections(lines):
    if len(lines) < 2:
        return []
    return [
        point
        for l1, l2 in combinations(lines, 2)
        if (point := line_intersection(l1, l2)) is not None
    ]


def remove_points_far_from_center(points, shape, dist=None):
    # TODO: IMPROVEMENT: remove outliers later, those far from the mean
    if dist is None:
        dist = shape[0] / 10
    return [
        p
        for p in points
        if (abs(p[1] - shape[0] / 2) < dist) and (abs(p[0] - shape[1] / 2) < dist)
    ]


def remove_outliers(points):
    mean, sd = np.mean(points, 0), np.std(points, 0)

    filtered = [
        p for p in points if (p[0] - mean[0] < 3 * sd[0] and p[1] - mean[1] < 1 * sd[1])
    ]
    return filtered


def remove_lines_of_similar_slope(lines, th=5):
    # TODO: IMPROVEMENT: select not the first but the longest out of similar lines
    if len(lines) < 1:
        return []
    filtered_lines = [lines[0]]
    for line in lines[1:]:
        similar = False
        for fline in filtered_lines:
            angle_diff = abs(get_angle(line) - get_angle(fline))
            if angle_diff < th:
                similar = True
                break
        if similar:
            continue
        filtered_lines.append(line)

    assert len(filtered_lines) <= len(lines)
    return filtered_lines
