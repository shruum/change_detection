import os
import sys

import numpy as np
from icecream import ic

from vpoint import localize_vanishing_point

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from segmentation import Segmenter


def baseline_vanishing_point(img_paths, segmenter):
    return 1920 // 2, 1080 // 2


def test_vpoint_single_images(localize):
    test_dict = {
        "./lanes/essence/0009.jpg": (911, 541),
        "./lanes/essence/0377.jpg": (902, 535),
        "./lanes/essence/2019-09-24192930535.jpg": (1016, 566),
        "./lanes/essence/2019-09-24193603028.jpg": (994, 567),
        "./lanes/essence/DE_20190701-121947-NF_00000078_7.07697780_51.46475458.jpg": (
            923,
            549,
        ),
        "./lanes/essence/DE_20190706-093902-NF_00003287_9.72283456_52.45964259.jpg": (
            921,
            545,
        ),
        "./lanes/essence/img_0168.jpg": (1024, 546),
        "./lanes/essence/img_0168b.jpg": (1023, 607),
        "./lanes/essence/rgb2.jpg": (915, 550),
        "./lanes/essence/rgb4.jpg": (913, 553),
    }

    segmenter = Segmenter("/tmp/bla/")
    ic(localize)
    for img_path, true_vpoint in test_dict.items():
        pred_vpoint = localize([img_path], segmenter)
        pred_vpoint = np.array(pred_vpoint)
        true_vpoint = np.array(true_vpoint)
        diff = np.linalg.norm(true_vpoint - pred_vpoint, 2)
        ic(diff)
        # assert diff < 10, f"{diff} < 15"


def test_vpoint_multi_images():
    test_dict = {
        "./lanes/essence/0009.jpg": (911, 541),
        "./lanes/essence/0377.jpg": (902, 535),
        "./lanes/essence/2019-09-24192930535.jpg": (1016, 566),
        "./lanes/essence/2019-09-24193603028.jpg": (994, 567),
        "./lanes/essence/DE_20190701-121947-NF_00000078_7.07697780_51.46475458.jpg": (
            923,
            549,
        ),
        "./lanes/essence/DE_20190706-093902-NF_00003287_9.72283456_52.45964259.jpg": (
            921,
            545,
        ),
        "./lanes/essence/img_0168.jpg": (1024, 546),
        "./lanes/essence/img_0168b.jpg": (1023, 607),
        "./lanes/essence/rgb2.jpg": (915, 550),
        "./lanes/essence/rgb4.jpg": (913, 553),
    }
    segmenter = Segmenter("/tmp/bla/")
    for hth in range(50, 200, 10):  # [50, 100, 150, 200]:
        d = []
        for img_path, true_vpoint in test_dict.items():
            # print("")
            # ic(img_path)
            # ic(true_vpoint)
            pred_vpoint = localize_vanishing_point([img_path], segmenter, hth)
            # ic(pred_vpoint)
            diff = np.linalg.norm(np.array(true_vpoint) - np.array(pred_vpoint), 2)
            # ic(diff)
            d.append(diff)
        print(hth, "\t", np.mean(d))


test_vpoint_single_images(baseline_vanishing_point)
test_vpoint_single_images(localize_vanishing_point)
