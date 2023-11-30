"""Line detector"""

import math
import cv2
import numpy as np

# DEBUG
# from icecream import ic
import matplotlib.pyplot as plt

plt.style.use("dark_background")

# TODO: 2d box, filter x-pos, y-frame
# TODO: dashed overrules solid?
# TODO: utilize angle of the edge detection (split between left right clusters)
#       especially for guardrails


def line_point_distance(line, point):
    x0, y0 = point
    x1, y1 = line[0], line[1]
    x2, y2 = line[2], line[3]
    dy = y2 - y1
    dx = x2 - x1
    return (dy * x0 - dx * y0 + x2 * y1 - y2 * x1) / math.sqrt(dy ** 2 + dx ** 2)


def line_length(line):
    return math.sqrt((line[3] - line[1]) ** 2 + (line[2] - line[0]) ** 2)


def line_slope(line):
    return (line[3] - line[1]) / (line[2] - line[0])


def line_intersection(line, line2):
    x1, y1, x2, y2 = line
    x3, y3, x4, y4 = line2
    dx12 = x1 - x2
    dx34 = x3 - x4
    dy12 = y1 - y2
    dy34 = y3 - y4
    denom = dx12 * dy34 - dy12 * dx34
    px = ((x1 * y2 - y1 * x2) * dx34 - dx12 * (x3 * y4 - y3 * x4)) / denom
    # NOTE: y-coordinate not needed
    return px


def plot_asterisk(img, center, size):
    angles = np.linspace(0.1, np.pi - 0.1, 12)
    for a in angles:
        a = math.tan(a)
        x1 = int(-size[1] / 2 / a)
        x2 = int(size[1] / 2 / a)
        if abs(x1) > abs(-size[0] / 2):
            x1 = int(-size[0] / 2)
            x2 = int(size[0] / 2)
        y1 = int(a * x1)
        y2 = int(a * x2)
        pt1 = (x1 + center[0], y1 + center[1])
        pt2 = (x2 + center[0], y2 + center[1])
        cv2.line(img, pt1, pt2, (255), 2)
    return img


def make_test_can(img_shape, img_dtype):
    CENTER = 1920 // 2 - 55, 1080 // 2 + 5
    img = np.zeros(img_shape, img_dtype)
    h, w = img.shape[:2]
    shrink = 2
    steps = np.array([-2, 0, 2])
    size = np.array([w // 3, h // 3], int)
    xsteps = steps * size[0] / shrink + CENTER[0]
    ysteps = steps * size[1] / shrink + CENTER[1]
    centers = np.array([[int(x), int(y)] for x in xsteps for y in ysteps])
    for center in centers:
        img = plot_asterisk(img, center, size)
    return img


class LaneDetector:
    def __init__(
        self,
        vpoint=(1920 // 2, 1080 // 2),
        top_clip=580,
        bot_clip=850,
        canny_th1=50,
        canny_th2=150,
        hough_rho=2,
        hough_theta=0.5 * np.pi / 180,
        hough_th=90,
        hough_minlen=100,
        hough_maxgap=100,
        vpoint_th=5,
        length_th=125,
        cluster_th=1,
    ):
        self.vpoint = vpoint
        # reference depth line for lane marking position
        self.ref_line = [
            self.vpoint[0] - 100,
            self.vpoint[1] + 100,
            self.vpoint[0] + 100,
            self.vpoint[1] + 100,
        ]
        self.top_clip = top_clip
        self.bot_clip = bot_clip
        self.canny_th1 = canny_th1
        self.canny_th2 = canny_th2
        self.hough_rho = hough_rho
        self.hough_theta = hough_theta
        self.hough_th = hough_th
        self.hough_minlen = hough_minlen
        self.hough_maxgap = hough_maxgap
        self.vpoint_th = vpoint_th
        self.length_th = length_th
        self.cluster_th = cluster_th

    def _crop(self, seg, top, bot):
        seg[:top, :] = 0
        seg[bot:, :] = 0
        return seg

    def _detect_edges(self, seg):
        # NOTE: blur doesn't help
        # seg = cv2.GaussianBlur(seg, (self.blur_ksize, self.blur_ksize), 0)
        can = cv2.Canny(seg, self.canny_th1, self.canny_th2)
        # can = make_test_can(seg.shape, seg.dtype)
        return can

    def _detect_lines(self, can):
        lines = cv2.HoughLinesP(
            can,
            rho=self.hough_rho,
            theta=self.hough_theta,
            threshold=self.hough_th,
            minLineLength=self.hough_minlen,
            maxLineGap=self.hough_maxgap,
        )
        if lines is None:
            return []
        return [np.squeeze(line) for line in lines]

    def _ab(self, line):
        a = line_slope(line)
        b = line[1] - a * line[0]
        return a, b

    def _filter_further_from_vpoint(self, lines):
        return [
            l for l in lines if line_point_distance(l, self.vpoint) < self.vpoint_th
        ]

    def _filter_shorter(self, lines):
        return [l for l in lines if line_length(l) > self.length_th]

    def _position(self, line):
        return line_intersection(line, self.ref_line) - self.vpoint[0]

    def detect(self, seg):
        seg_crop = self._crop(seg, self.top_clip, self.bot_clip)
        can = self._detect_edges(seg_crop)
        # can = self._crop(can, self.top_clip + 5, self.bot_clip)
        lines = self._detect_lines(can)
        lines = self._filter_further_from_vpoint(lines)
        lines = self._filter_shorter(lines)
        xs = [self._position(l) for l in lines if l[1] != l[3]]
        # cluster
        return xs

    def detect_debug(self, seg, debug=False):
        seg_crop = self._crop(seg, self.top_clip, self.bot_clip)
        can = self._detect_edges(seg_crop)
        lines = self._detect_lines(can)
        all_lines = lines.copy()
        lines = self._filter_further_from_vpoint(lines)
        centered_lines = lines.copy()
        lines = self._filter_shorter(lines)
        long_centered_lines = lines.copy()
        xs = [self._position(l) for l in lines if l[1] != l[3]]
        return (
            can,
            all_lines,
            centered_lines,
            long_centered_lines,
            xs,
        )
