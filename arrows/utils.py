import sys
import numpy as np
import cv2
from scipy import fftpack


def binarize(seg_map, target_class):
    """

    :param seg_map: A segmentation map with pixel value = class labels
    :param target_class: Target class for making binary map
    :return: Binary map which is True wherever target class is present
    """
    return 255 * (seg_map == target_class)


def smooth_count(counts):
    """

    :param counts: list of counts of detection
    :return: Smoothed list of counts, where anomalies are replaced with neighbours
    """
    n = len(counts)
    for i in range(n - 1):
        if i >= 1 and counts[i - 1] == counts[i + 1] and counts[i] != counts[i + 1]:
            counts[i] = counts[i - 1]
        if (
            1 <= i < n - 2
            and counts[i - 1] == counts[i + 2]
            and counts[i] != counts[i + 2]
        ):
            counts[i] = counts[i - 1]
            counts[i + 1] = counts[i - 1]
    return counts


def prune_contours(contours, thresh=50):
    """

    :param contours:Contours to prune
    :param thresh: Minimum area of contours
    :return: Pruned list of contours
    """
    pruned_contours = []
    for contour in contours:
        if cv2.contourArea(contour) > thresh:
            pruned_contours.append(contour)
    return pruned_contours


def get_shifts(counts_run1, counts_run2):
    """

    :param counts_run1: list of per-frame detection counts from run 1
    :param counts_run2: list of per-frame detection counts from run 2
    :return: time-shift between lists in both directions
    """
    A = fftpack.fft(counts_run1)
    B = fftpack.fft(counts_run2)
    Ar = -A.conjugate()
    Br = -B.conjugate()
    left_shift = np.argmax(np.abs(fftpack.ifft(Ar * B)))
    right_shift = np.argmax(np.abs(fftpack.ifft(A * Br)))
    return left_shift, right_shift


def change_type(diff, mode):
    """

    :param diff: Difference in counts of detection
    :param mode: left or right denoting direction in which change is being checked
    :return: status of change
    """
    if mode == "left":
        x = "Removed" if diff > 0 else "Added"
    else:
        x = "Removed" if diff < 0 else "Added"
    return x


def set_index(index, upper_limit=sys.maxsize, lower_limit=0):
    if index < lower_limit:
        return 0
    elif index > upper_limit:
        return upper_limit
    else:
        return index


def process_change(change_status, frame_start, frame_end):
    change = [
        change_status,
        "Arrow",
        "",
        "",
        "change_detected",
        "",
        1,
        "",
        "",
        "",
        "",
        "",
        "",
        "CRDv2",
        "",
        "",
        "",
        frame_start,
        frame_end,
    ]
    return change
