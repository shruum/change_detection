from typing import List
import logging

import os
import numpy as np

from utils import imread
from lanes.detector import LaneDetector

from tqdm import tqdm

LocFrameCls = np.dtype([("location", "f4"), ("frame", "i4"), ("class", "i4")])


def detect_lane_changes(run1_paths, run2_paths, segmenter, vpoints, cache_dir) -> List:
    logging.info("Detecting Lane Change...")
    lane_cd = LaneChangeDetector(segmenter, [2, 1], vpoints, cache_dir)
    changes = lane_cd.detect_changes(run1_paths, run2_paths)
    return changes


class LaneChangeDetector:
    def __init__(self, segmenter, class_ids, vpoints, cache_dir):
        self.detector = LaneDetector(vpoints[0])
        self.detector_b = LaneDetector(vpoints[1])
        self.segmenter = segmenter
        self.class_ids = class_ids
        self.diff_threshold = 200
        self.cache_dir = os.path.join(cache_dir, "lanes")

    def detect_changes(self, rgb_paths_a, rgb_paths_b):
        logging.info("detecting lane boundary changes...")
        detector = self.detector
        detector_b = self.detector

        data_file = os.path.join(self.cache_dir, "data_a.npy")
        if not os.path.exists(data_file):
            data = detect_fast(
                rgb_paths_a, self.segmenter, self.detector, self.class_ids
            )
            np.save(data_file, data)
            logging.info(f"Cached {data_file}")
        logging.info(f"Reading from cached {data_file}")
        data_a = np.load(data_file)

        data_file = os.path.join(self.cache_dir, "data_b.npy")
        if not os.path.exists(data_file):
            data = detect_fast(
                rgb_paths_b, self.segmenter, self.detector_b, self.class_ids
            )
            np.save(data_file, data)
            logging.info(f"Cached {data_file}")
        logging.info(f"Reading from cached {data_file}")
        data_b = np.load(data_file)

        data_ap = process(data_a)
        data_bp = process(data_b)

        data_diff = compare(data_ap, data_bp)
        changes, data_gated = gate(data_diff, self.diff_threshold)
        logging.info("lane boundary changes detected.")
        return changes
        # f = mpl_show_data_twin([data_ap, data_bp, data_diff, data_gated])
        # plt.savefig(f"./lanes/full.png")
        # plots = bokeh_show_data_twin([data_ap, data_bp, data_diff, data_gated])
        # bokeh.io.show(plots)


def apply_custom_map(im_gray, cls):
    lut = np.zeros((256, 1, 1), dtype=np.uint8)
    lut[cls, 0, 0] = 255  # white
    im_color = cv2.LUT(im_gray, lut)
    return im_color


def detect_class_lines(seg, detector: LaneDetector, class_ids):
    return {cls: detector.detect(apply_custom_map(seg, cls)) for cls in class_ids}


def detect_fast(rgb_paths, segmenter, detector, class_ids):
    data = []
    for i, rgb_path in enumerate(tqdm(rgb_paths)):
        logging.debug(rgb_path)
        seg = segmenter.segment(rgb_path)
        lanes = detect_class_lines(seg, detector, class_ids)
        for k, v in lanes.items():
            for x in v:
                data.append((x, i, k))
    data = np.array(data, dtype=LocFrameCls)
    return data


def cluster_all_within_range(data, range_=100):
    """input data should be sorted"""
    grs = [[data[0]]]
    for point in data[1:]:
        if point[0] - grs[-1][0][0] < range_:
            grs[-1].append(point)
        else:
            grs.append([point])
    return grs


def cluster_within_range_assign_plurality(data):
    sorted_data = sorted(data, key=lambda x: x[0])

    def plurality(g):
        classes = [x[2] for x in g]
        unique_classes, counts = np.unique(classes, return_counts=True)
        return unique_classes[np.argmax(counts)]

    grs = cluster_all_within_range(sorted_data)
    clustered = [
        (np.mean([x[0] for x in g], 0), int(g[0][1]), plurality(g)) for g in grs
    ]
    return clustered


def cluster_within_range_per_class(data):
    """cluster point for each class separately"""
    unique_classes, counts = np.unique(data["class"], return_counts=True)
    clustered = []
    for cls in unique_classes.astype(int):
        cls_data = data[data["class"] == cls]
        sorted_cls_data = cls_data[cls_data["location"].argsort()]
        grs = cluster_all_within_range(sorted_cls_data)
        clustered += [(np.mean([x[0] for x in g], 0), int(g[0][1]), cls) for g in grs]
    return clustered


def process(data, span=0, cluster_type="per_class"):
    """Cluster the line segment detections, to have single point per lane marking.

    span +/- number of frames"""
    processed = []
    frame_ids = sorted(list(set(data["frame"])))
    frame_span = frame_ids if span == 0 else frame_ids[span:-span]
    for i in frame_span:
        data_ = data[i - span <= data["frame"]]
        frame_data = data_[data_["frame"] <= i + span]
        if cluster_type == "all":
            new_frame_data = cluster_within_range_assign_plurality(frame_data)
        elif cluster_type == "per_class":
            new_frame_data = cluster_within_range_per_class(frame_data)
        else:
            assert False, "not supported clustering"
        processed += new_frame_data
    return np.array(processed, dtype=LocFrameCls)


def arr_to_str(frame_data):
    sorted_frame_data = sorted(frame_data, key=lambda x: x[0])
    lanes = [int(x[2]) for x in sorted_frame_data]
    return "".join([str(x) for x in lanes if x in {1, 2}])


def compare(before, after):
    diff = []
    frame_ids = sorted(list(set(before["frame"])))
    counter = 0
    for i in frame_ids:
        frame_a = before[before["frame"] == i]
        frame_b = after[after["frame"] == i]
        lanes_a = arr_to_str(frame_a)
        lanes_b = arr_to_str(frame_b)
        if lanes_a == lanes_b:
            counter = 0
            diff.append(0)
        else:
            counter += 1
            diff.append(counter)

    return np.array(diff)


def compare2(before, after):
    """a bit more robust to occlusions - maybe"""
    diff = []
    frame_ids = [int(x) for x in sorted(set(before[:, 1]))]
    counter = 0
    for i in frame_ids:
        frame_a = before[before[:, 1] == i]
        frame_b = after[after[:, 1] == i]
        lanes_a = arr_to_str(frame_a)
        lanes_b = arr_to_str(frame_b)
        if (lanes_a in lanes_b) or (lanes_b in lanes_a):
            counter = 0
            diff.append(0)
        else:
            counter += 1
            diff.append(counter)
    return np.array(diff)


def gate(diff, threshold=200):
    out = []
    change = False
    changes = []
    counter_prev = 0
    for i, counter in reversed(list(enumerate(diff))):
        if counter_prev < counter:
            if counter > threshold:
                change = True
                print("change end:", i)
                changes.append([0, i])
        if counter == 0:
            if change:
                change = False
                print("change start:", i)
                changes[-1][0] = i
        if change:
            out.append(1)
        else:
            out.append(0)
        counter_prev = counter
        print("counter", counter)
    return list(reversed(changes)), np.array(out)
