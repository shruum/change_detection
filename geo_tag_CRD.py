import pathlib
import sys

import cv2
import numpy as np
import textdistance
from PIL import Image
import argparse
import os
import tqdm

from text_reader.read_and_locate import Config, read_crop


def get_center(cnt):
    M = cv2.moments(cnt)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    return cX, cY


def smoke_break(pts1, pts2):
    """
    Gets distance between two points
    :param pts1:
    :param pts2:
    :return:
    """
    return np.linalg.norm([pts1, pts2])


def get_cnt_area(cnt):
    return cv2.contourArea(cnt)


def cnt_to_boxes(contours, thresh=20000, y_thresh=100):
    box_cnts = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h > thresh and y <= y_thresh:
            box_cnts.append(
                np.array([(x, y), (x + w, y), (x + w, y + h), (x, y + h)])[
                    :, np.newaxis, :
                ]
            )
    return box_cnts


def cnt_to_crops(image, contours, thresh=20000):
    box_crops = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h > thresh:
            temp = cv2.cvtColor(image[y : y + h, x : x + w], cv2.COLOR_BGR2RGB)
            box_crops.append(Image.fromarray(temp))
    return box_crops


def pair_strings(strings_1, strings_2, thresh=0.8):
    pair_list = []
    for string_1 in strings_1:
        distances = np.array(
            [
                textdistance.ratcliff_obershelp(string_1, string_2)
                for string_2 in strings_2
            ]
        )
        max_sim = np.argmax(distances)
        max_dist = np.max(distances)
        if max_dist > thresh:
            pair = string_1 + " -> " + strings_2[max_sim]
            del strings_2[max_sim]
        else:
            pair = string_1 + " -> Removed"
        pair_list.append(pair)
    for string_2 in strings_2:
        pair_list.append((string_2 + " -> New Value"))
    return np.array(pair_list)


conf = Config()


def write_dets_npy(engine, text_file, save_dir="./text_detections/", npy_name="run"):
    os.makedirs(save_dir, exist_ok=True)
    text_info = open(text_file, "r")
    info = np.array(
        [line.split(",") for line in text_info.readlines() if "GUID," not in line]
    )
    img_list = info[:, 1]
    final_array = None
    for n in tqdm.tqdm(range(len(img_list))):
        lines, confs, cents = [], [], []
        img_pth = img_list[n]
        lab = np.array(engine.segment(img_pth))
        lab = (lab == 4) / 1
        tmp = (255 * lab[..., np.newaxis]).astype("uint8")
        cnts, _ = cv2.findContours(tmp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(cnts) > 1:
            img = cv2.imread(img_pth)
            crops = cnt_to_crops(img, cnts)
            if len(crops):
                for crop in crops:
                    try:
                        curr_lines, curr_confs, curr_centers = read_crop(crop, conf)
                    except ValueError:
                        continue
                    formatted_lines = [line.split("[s]", 1)[0] for line in curr_lines]
                    lines.append(formatted_lines)
                    confs.append(curr_confs)
                    cents.append(curr_centers)

                flat_list = [item for sublist in lines for item in sublist]
                if len(flat_list):
                    coord = info[n, 17:20]
                    if final_array is None:
                        final_array = np.array(
                            [
                                img_pth,
                                (float(coord[0]), float(coord[1])),
                                lines,
                                confs,
                                cents,
                                coord[-1],
                            ]
                        )[np.newaxis]
                    else:
                        curr_array = np.array(
                            [
                                img_pth,
                                (float(coord[0]), float(coord[1])),
                                lines,
                                confs,
                                cents,
                                coord[-1],
                            ]
                        )[np.newaxis]
                        final_array = np.concatenate([final_array, curr_array], axis=0)
            else:
                pass
        else:
            pass
    final_save_pth = os.path.join(save_dir, npy_name)
    np.save(final_save_pth, final_array)


# engine = Segmenter("./")
# =============================================== #
# file_obj.writelines(final_list)
# file_obj.close()
# =============================================== #
