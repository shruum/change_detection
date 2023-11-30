from arrows.utils import binarize
import numpy as np
import cv2
import tqdm
import logging
import os
import sys
from utils import imread, imwrite

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_PATH, "..", "nie_utilities"))
from data_processing_utils import create_nie_json, get_basename_no_ext, save_json


class ArrowDetector:
    def __init__(
        self,
        segmenter,
        target_class,
        confounding_class,
        confounding_border_thresh,
        thresh,
        dilation_size,
        min_cnt_dist,
        write_output=False,
    ):
        """

        :param segmenter: Segmentation model generating 2d numpy array of image size with class labels as pixel values
        :param target_class: The class to which arrow belongs in the segmentation output
        :param confounding_class: The class which arrow segmentation gives false positive for
        :param confounding_border_thresh: Border around false positive detection to remove detections
        :param thresh: The minimum area of bounding box of contour for it to be considered a valid detection
        :param dilation_size: Size of dilation kernel used on contours obtained from binarized segmentation map
        :param min_cnt_dist: Min. separation at which contours are considered separate

        """
        self.segmenter = segmenter
        self.target_class = target_class
        self.thresh = thresh
        self.dilation_size = dilation_size
        self.min_cnt_dist = min_cnt_dist
        self.write_output = write_output
        self.confounding_class = confounding_class
        self.confounding_border_thresh = confounding_border_thresh
        if write_output:
            self.cache_dir = os.path.join(
                self.segmenter.cache_dir.replace("segmentation", ""), "arrow_detections"
            )
            if not os.path.exists(self.cache_dir):
                os.mkdir(self.cache_dir)
                os.mkdir(os.path.join(self.cache_dir, "run1"))
                os.mkdir(os.path.join(self.cache_dir, "run2"))

    def detect_and_count(self, run1_paths, run2_paths):
        """

        :param run1_paths: ordered image paths from aligned run 1
        :param run2_paths: corresponding image paths from aligned run 2
        :return: counts of detections for each frame in each run
        """
        counts_l = []
        counts_r = []
        logging.info("detecting arrows...")
        for path1, path2 in tqdm.tqdm(
            zip(run1_paths, run2_paths), total=len(run1_paths)
        ):
            map1 = binarize(self.segmenter.segment(path1), self.target_class)
            map2 = binarize(self.segmenter.segment(path2), self.target_class)
            map3 = binarize(self.segmenter.segment(path1), self.confounding_class)
            map4 = binarize(self.segmenter.segment(path2), self.confounding_class)
            detections = self.detect_aligned(map1, map2, map3, map4)
            counts_l.append(len(detections[0]))
            counts_r.append(len(detections[1]))
            if self.write_output:
                self.save_detections(path1, "run1", detections[0])
                self.save_detections(path2, "run2", detections[1])
        return counts_l, counts_r

    def detect_aligned(self, map1, map2, map3, map4):
        """

        :param map1: binary map of target class segmentation from run 1 frame
        :param map2: binary map of target class segmentation from run 2 frame
        :param map3: binary map of confounding class segmentation from run 1 frame
        :param map4: binary map of confounding class segmentation from run 2 frame
        :return: number of bounding boxes/objects detected in each run
        """
        s1 = np.sum(map1)
        s2 = np.sum(map2)
        bboxes3, bboxes4 = [], []

        if s1 > 0:
            s3 = np.sum(map3)
            bboxes3 = self.detect(map3) if s1 * s3 != 0 else []

        if s2 > 0:
            s4 = np.sum(map4)
            bboxes4 = self.detect(map4) if s2 * s4 != 0 else []

        if bboxes3:
            map3 = map3 * 0 + 1
            for bbox in bboxes3:
                x, y, w, h = bbox
                map3[
                    max(y - self.confounding_border_thresh, 0) : min(
                        y + h + self.confounding_border_thresh, map3.shape[0]
                    ),
                    max(x - self.confounding_border_thresh, 0) : min(
                        x + w + self.confounding_border_thresh, map3.shape[1]
                    ),
                ] = 0
            map1 *= map3

        if bboxes4:
            map4 = map4 * 0 + 1
            for bbox in bboxes4:
                x, y, w, h = bbox
                map4[
                    max(y - self.confounding_border_thresh, 0) : min(
                        y + h + self.confounding_border_thresh, map4.shape[0]
                    ),
                    max(x - self.confounding_border_thresh, 0) : min(
                        x + w + self.confounding_border_thresh, map4.shape[1]
                    ),
                ] = 0
            map2 *= map4

        s1 = np.sum(map1)
        s2 = np.sum(map2)

        bboxes1 = self.detect(map1) if s1 != 0 else []
        bboxes2 = self.detect(map2) if s2 != 0 else []

        return bboxes1, bboxes2

    def detect(self, map):
        """

        :param map: binary map of target class segmentation for a frame
        :return: bounding boxes of detections
        """
        kernel = np.ones((self.dilation_size, self.dilation_size), np.uint8)
        map = cv2.dilate(map.astype(np.uint8), kernel, iterations=1)
        contours, hierarchy = cv2.findContours(
            map.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        cnts = self.merge_contours(contours)
        bboxes = self.cnt_to_boxes(cnts)
        # bboxes = prune_contours(cnts, thresh=thresh)
        return bboxes

    def merge_contours(self, contours):
        """

        :param contours: contours extracted from a frame
        :return: list of contours after merging contours close together
        """
        length_cnts = len(contours)
        status = np.zeros((length_cnts, 1))

        for i, cnt1 in enumerate(contours):
            x = i
            if i != length_cnts - 1:
                for j, cnt2 in enumerate(contours[i + 1 :]):
                    x = x + 1
                    dist = self.find_if_close(cnt1, cnt2)
                    if dist:
                        val = min(status[i], status[x])
                        status[x] = status[i] = val
                    else:
                        if status[x] == status[i]:
                            status[x] = i + 1
        unified = []
        maximum = int(status.max()) + 1
        for i in range(maximum):
            pos = np.where(status == i)[0]
            if pos.size != 0:
                ip = [contours[i] for i in pos]
                cont = np.vstack(ip)
                hull = cv2.convexHull(cont)
                unified.append(hull)
        return unified

    def find_if_close(self, cnt1, cnt2):
        """

        :return: True if contours are closer than min_cnt_dist, else False
        """
        row1, row2 = cnt1.shape[0], cnt2.shape[0]
        for i in range(row1):
            for j in range(row2):
                dist = np.linalg.norm(cnt1[i] - cnt2[j])
                if abs(dist) < self.min_cnt_dist:
                    return True
                elif i == row1 - 1 and j == row2 - 1:
                    return False

    def cnt_to_boxes(self, contours):
        """

        :param contours: List of contours
        :return: bounding boxes for contours
        """
        box_cnts = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w * h > self.thresh:
                box_cnts.append((x, y, w, h))
        return box_cnts

    def save_detections(self, img_path, run_name, detections):
        if len(detections) == 0:
            return
        detections_to_save = []
        image = cv2.imread(img_path)
        img_height = image.shape[0]
        img_width = image.shape[1]
        for detection in detections:
            x, y, w, h = detection
            detections_to_save.append([y, x, h, w, "arrow", 8060])
        json_dict = create_nie_json(img_width, img_height, detections_to_save)
        output_path = os.path.join(
            self.cache_dir, run_name, get_basename_no_ext(img_path) + ".json"
        )
        save_json(json_dict, output_path)
