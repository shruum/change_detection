import os

import cv2
import numpy as np
from icecream import ic

from utils import imread, imwrite

from lanes.utils import (
    apply_custom_color_map,
    my_color_rgb,
    my_white_rgb,
    get_angle,
    intersections,
    remove_outliers,
    remove_points_far_from_center,
    remove_lines_of_similar_slope,
)
from lanes.detector import LaneDetector


def prep(img):
    if len(img.shape) < 3:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return img


def draw_lines(lines, img_shape, img_dtype, color=(255, 255, 255)):
    img = np.zeros(img_shape, img_dtype)
    for i, line in enumerate(lines):
        cv2.line(img, (line[0], line[1]), (line[2], line[3]), color, 2)
    return img


def draw_points(points, img, ch=0):
    for point in points:
        if 0 < point[0] < img.shape[0]:
            if 0 < point[1] < img.shape[1]:
                COLOR_BOOST = 100
                img[int(point[1]), int(point[0])][ch] = min(
                    img[int(point[1]), int(point[0])][ch] + COLOR_BOOST, 255
                )
    return img


def draw_ellipse(img, point, radius=5, ch=0):
    img = np.array(img).astype(np.uint8)
    ic((int(point[0]), int(point[1])))
    ic((int(radius[0]), int(radius[1])))
    cv2.ellipse(
        img,
        (int(point[0]), int(point[1])),
        (int(radius[0]), int(radius[1])),
        angle=0,
        startAngle=0,
        endAngle=360,
        color=(255, 0, 255),
        thickness=2,
    )
    return img


def detect_and_draw_lines(detector, can):
    lines = detector._detect_lines(can)
    lines_img_raw = draw_lines(lines, (*can.shape, 3), can.dtype, my_color_rgb(2))
    lines = remove_lines_of_similar_slope(lines)
    lines_img = draw_lines(lines, (*can.shape, 3), can.dtype, my_color_rgb(1))
    return lines, lines_img, lines_img_raw


def draw_intersections(lines, img):
    img = img.copy()
    points = intersections(lines)
    points = remove_points_far_from_center(points, img.shape)

    vpoint = 0, 0
    if points:
        img = draw_points(points, img)
        vpoint, vpoint_std = np.mean(points, 0), np.std(points, 0)
        img[int(vpoint[1]), int(vpoint[0])][0] = 0
        img[int(vpoint[1]), int(vpoint[0])][1] = 255
        img[int(vpoint[1]), int(vpoint[0])][2] = 0
        points = remove_outliers(points)
        points = draw_points(points, img, 2)
    return img, vpoint


def main(img_paths, segmenter):
    def nothing(x):
        pass

    img = imread(img_paths[0])
    seg = segmenter.segment(img_paths[0])
    seg3 = cv2.cvtColor(seg, cv2.COLOR_GRAY2RGB)
    seg3w = apply_custom_color_map(seg3, [2, 1, 9], my_white_rgb)
    seg1w = cv2.cvtColor(seg3w, cv2.COLOR_RGB2GRAY)
    detector = LaneDetector()

    cv2.namedWindow("window")
    canny_th1 = detector.canny_th1
    canny_th2 = detector.canny_th2
    hough_rho = detector.hough_rho
    hough_theta = int(detector.hough_theta * 180.0 / np.pi * 10)
    hough_th = detector.hough_th
    hough_maxgap = detector.hough_maxgap
    dil_size = 3
    cv2.createTrackbar("canny_th1", "window", canny_th1, 255, nothing)
    cv2.createTrackbar("canny_th2", "window", canny_th2, 255, nothing)
    cv2.createTrackbar("hough_rho", "window", hough_rho, 255, nothing)
    cv2.createTrackbar("hough_theta", "window", hough_theta, 255, nothing)
    cv2.createTrackbar("hough_th", "window", hough_th, 255, nothing)
    cv2.createTrackbar("hough_maxgap", "window", hough_maxgap, 255, nothing)
    cv2.createTrackbar("dil_size", "window", dil_size, 255, nothing)

    while True:
        can = cv2.Canny(img, canny_th1, canny_th2)
        canseg = cv2.Canny(seg3w, canny_th1, canny_th2)

        element = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (dil_size * 2, dil_size * 2), (dil_size, dil_size)
        )
        seg1wf = cv2.dilate(seg1w, element)

        # can_x_seg1w = cv2.bitwise_and(can, seg1w)
        can_x_seg1wf = cv2.bitwise_and(can, seg1wf)

        # lines1, lines_img1 = detect_and_draw_lines(detector, can)
        # lines2, lines_img2 = detect_and_draw_lines(detector, canseg)
        # lines3, lines_img3 = detect_and_draw_lines(detector, can_x_seg1w)
        lines4, lines_img4, lines_img4_raw = detect_and_draw_lines(
            detector, can_x_seg1wf
        )

        # points_img1, vpoint1 = draw_intersections(lines1, lines_img1)
        # points_img2, vpoint2 = draw_intersections(lines2, lines_img2)
        # points_img3, vpoint3 = draw_intersections(lines3, lines_img3)
        points_img4, vpoint4 = draw_intersections(lines4, lines_img4)

        # NOTE: if the tuner is too slow, set flag below
        simple_fast_variant = False
        if simple_fast_variant:
            cv2.imshow(
                "window", np.vstack([prep(img)[:, :, ::-1], prep(lines_img4_raw)])
            )
        else:
            cv2.imshow(
                "window",
                np.hstack(
                    [
                        np.vstack([prep(img)[:, :, ::-1], prep(seg1w)]),
                        np.vstack([prep(can), prep(can_x_seg1wf)]),
                        np.vstack(
                            [
                                prep(lines_img4_raw),
                                # prep(points_img2),
                                # prep(points_img3),
                                prep(points_img4),
                            ]
                        ),
                    ]
                ),
            )

        k = cv2.waitKey(1) & 0xFF
        if k == 27 or k == 113:  # esc or q
            break
        canny_th1 = cv2.getTrackbarPos("canny_th1", "window")
        canny_th2 = cv2.getTrackbarPos("canny_th2", "window")
        detector.hough_rho = cv2.getTrackbarPos("hough_rho", "window")
        detector.hough_theta = (
            cv2.getTrackbarPos("hough_theta", "window") * np.pi / 180 / 10
        )
        detector.hough_th = cv2.getTrackbarPos("hough_th", "window")
        detector.hough_maxgap = cv2.getTrackbarPos("hough_maxgap", "window")
        dil_size = cv2.getTrackbarPos("dil_size", "window")

    cv2.destroyAllWindows()


def dbg_img(img, name):
    os.makedirs("./cache/debug/vpoint/", exist_ok=True)
    imwrite(f"./cache/debug/vpoint/{name}.png", img)


def vpoint_vis1(img, seg, seg3w, seg3, can, seg1wf, can_x_seg1wf):
    dbg_img(img, "01_img")
    dbg_img(seg, "02_seg")
    dbg_img(seg3w, "03_seg3w")
    dbg_img(apply_custom_color_map(seg3, [2, 1], my_color_rgb), "03_seg3c")
    dbg_img(can, "04_can")
    dbg_img(seg1wf, "05_seg1wf")
    dbg_img(can_x_seg1wf, "06_can_x_seg1wf")


def vpoint_vis1b(
    img, lines, unique_lines, points, center_points,
):
    ic(lines)
    ic(unique_lines)
    ic(points)
    ic(center_points)
    lines_angles = [(l, get_angle(l)) for l in lines]
    ic(lines_angles)
    unique_lines_angles = [(l, get_angle(l)) for l in unique_lines]
    ic(unique_lines_angles)
    dbg_img(
        draw_lines(lines, img.shape, img.dtype, my_white_rgb(1)), "07_detected_lines",
    )
    dbg_img(
        draw_lines(unique_lines, img.shape, img.dtype, my_white_rgb(1)),
        "08_unique_lines",
    )
    empty_img = np.zeros(img.shape, img.dtype)
    dbg_img(draw_points(points, empty_img), "09_points")


def vpoint_vis2(img, center_points, point, point_std):
    empty_img = np.zeros(img.shape, img.dtype)
    dbg_img(draw_points(center_points, empty_img), "10_center_points")

    ic(point)
    ic(point_std)

    dbg_img(draw_ellipse(img.copy(), point, point_std, 1), "11_vpoint")
    ic(int(point[0]), int(point[1]))


def vpoint_vis3(img_path, vpoint, vpoint_std):
    dbg_img(draw_ellipse(imread(img_path), vpoint, vpoint_std, 1), "vpoint")
