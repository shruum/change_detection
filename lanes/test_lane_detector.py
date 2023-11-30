"""Line detector test

I recommend running mode0 first, checking if the line are detected fine,
if not adjust the vanishing point (VPOINT) location.
Then run mode2
if yes, move to mode3.
"""

import argparse
import os
import shutil
from collections import defaultdict, namedtuple
from datetime import timedelta
from glob import glob
from time import time


import cv2
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# plt.style.use("dark_background")
plt.style.use("fast")

import bokeh.io
import bokeh.plotting
import bokeh.layouts

bokeh.io.output_file("lanes.html")
# bokeh.io.curdoc().theme = "dark_minimal"


from .detector import LaneDetector
from .change_detector import (
    apply_custom_map,
    detect_class_lines,
    detect_fast,
    process,
    compare,
    gate,
)
from segmentation import Segmenter
from lanes.utils import my_color_name, my_color_rgb, apply_custom_color_map

segmenter = Segmenter("/tmp/")

from utils import imread, imwrite


### GLOBAL CONSTANTS ###

VPOINT = None  # TODO: find center automatically - hough of hough?
VPOINT_B = None  # TODO: find center automatically - hough of hough?


### TYPES ###

HParams = namedtuple("HParams", ["dataset", "mode", "diff_threshold", "ylimit"])

### FUNCTIONS ###


def get_args():
    global VPOINT
    global VPOINT_B

    parser = argparse.ArgumentParser(
        description="Customize test scriptGet inputs.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        choices=["augsburg", "augsburg2", "china1video"],
        default="augsburg",
        help="Testing dataset",
    )
    parser.add_argument(
        "--mode",
        choices=["mode0", "mode1", "mode2", "mode3", "mode4"],
        default="mode3",
        help="""mode0 - visualize_detection_stages
mode1 - print_lane_locations_and_plot
mode2 - show/save 2 rgbs image and 2 det plots below, on the fly for each pair
mode3 - if not cached, detect all-through run1, then run2, then cache it,
        then show 2 detection plots, diff and change side-by-side;
mode4 - create images for video, very slow (naive code)
""",
    )
    args = parser.parse_args()

    # if args.dataset == "augsburg":
    VPOINT = 1920 // 2 - 55, 1080 // 2 + 5
    VPOINT_B = VPOINT
    diff_threshold = 75
    ylimit = 300
    if args.dataset == "augsburg2":
        VPOINT = 1920 // 2 - 55, 1080 // 2 + 5
        VPOINT_B = VPOINT
        diff_threshold = 200
        ylimit = 1000
    if args.dataset == "china1video":
        VPOINT = 1020, 542
        VPOINT_B = 1024, 604
        diff_threshold = 200  # TODO: adjust (depends on fps)
    if args.dataset == "germany":
        VPOINT = 1000, 620
        VPOINT_B = VPOINT
        diff_threshold = 200  # TODO: adjust (depends on fps)

    return HParams(args.dataset, args.mode, diff_threshold, ylimit)


def label2rgb(labels):
    """Convert a labels image to an rgb image using a matplotlib colormap"""
    label_range = np.linspace(0, 1, 256)
    # replace viridis with a matplotlib colormap of your choice
    lut = np.uint8(plt.cm.rainbow(label_range)[:, 2::-1] * 255).reshape(256, 1, 3)
    return cv2.LUT(cv2.merge((labels, labels, labels)), lut)


def _conv_if_needed(img):
    if len(img.shape) > 2:
        if img.shape[2] == 3:
            return img
    return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)


def stack_imshow(*args):
    data = np.vstack([np.hstack([_conv_if_needed(img) for img in arg]) for arg in args])
    plt.imshow(data, aspect="auto")
    plt.tight_layout()
    plt.show()


def gridspec_imshow(*args):
    rows = [[_conv_if_needed(img) for img in arg] for arg in args]
    fig = plt.figure(figsize=(16 * 8, 9 * 3))
    spec = gridspec.GridSpec(len(rows), len(rows[0]), figure=fig)
    for i, row in enumerate(rows):
        for j, img in enumerate(row):
            ax = fig.add_subplot(spec[i, j])
            ax.imshow(img)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            # ax.tick_params(axis='x', bottom=False)
            # imwrite(f"{i}{j}.png", img)
    plt.tight_layout()
    fig.patch.set_facecolor("white")
    plt.show()


def draw_lines(lines, img_shape, img_dtype, VPOINT, color=(255, 255, 255)):
    img = np.zeros(img_shape, img_dtype)
    for i, line in enumerate(lines):
        color = tuple(int(c) for c in (np.array(color) + (i - 5) * 3))
        cv2.line(img, (line[0], line[1]), (line[2], line[3]), color, 2)
    cv2.circle(img, (VPOINT[0], VPOINT[1]), 5, (255, 255, 255), 2)
    # cv2.line(
    #     img,
    #     (VPOINT[0] - 900, VPOINT[1] + 100),
    #     (VPOINT[0] + 900, VPOINT[1] + 100),
    #     (255, 255, 255),
    #     2,
    # )
    return img


def draw_lines_x(lines, img_shape, img_dtype, VPOINT, color=(255, 255, 255)):
    img = np.zeros(img_shape, img_dtype)
    for i, line in enumerate(lines):
        color = tuple(int(c) for c in (np.array(color) + (i - 5) * 3))
        cv2.rectangle(
            img,
            (VPOINT[0] + int(line) - 3, VPOINT[1] + 100 - 6),
            (VPOINT[0] + int(line) + 3, VPOINT[1] + 100 + 6),
            color,
            -1,
        )
    cv2.circle(img, (VPOINT[0], VPOINT[1]), 5, (255, 255, 255), 2)
    cv2.line(
        img,
        (VPOINT[0] - 900, VPOINT[1] + 100),
        (VPOINT[0] + 900, VPOINT[1] + 100),
        (255, 255, 255),
        1,
    )
    return img


def visualize_detection_stages(rgb_path, detector, class_ids):
    rgb = imread(rgb_path)
    blank = np.zeros_like(rgb)
    shape, dtype = rgb.shape, rgb.dtype

    seg_path = segmenter.segment(rgb_path)
    seg = imread(seg_path)
    seg3 = cv2.cvtColor(seg, cv2.COLOR_GRAY2RGB)
    seg3 = apply_custom_color_map(seg3, class_ids)

    debug_imgs = defaultdict(list)
    for cls in class_ids:
        seg_1class = apply_custom_map(seg, cls)
        (can, all_lines, centered_lines, long_centered_lines, xs) = detector.detect(
            seg_1class, debug=True
        )
        debug_imgs["seg"].append(seg_1class)
        debug_imgs["can"].append(can)
        debug_imgs["det"].append(
            draw_lines(all_lines, shape, dtype, VPOINT, my_color_rgb(cls))
        )
        debug_imgs["cen"].append(
            draw_lines(centered_lines, shape, dtype, VPOINT, my_color_rgb(cls))
        )
        debug_imgs["fil"].append(
            draw_lines(long_centered_lines, shape, dtype, VPOINT, my_color_rgb(cls))
        )
        debug_imgs["clu"].append(
            draw_lines_x(xs, shape, dtype, VPOINT, my_color_rgb(cls))
        )

    table_of_images = [
        [
            rgb if i == 0 else (seg3 if i == 1 else blank),
            debug_imgs["seg"][i],  # class segmentation
            debug_imgs["can"][i],  # edges
            # debug_imgs["det"][i],  # hough
            # debug_imgs["cen"][i],  # centered
            debug_imgs["fil"][i],  # long centered
            # debug_imgs["clu"][i],  # clustered
            (rgb // 2 + debug_imgs["clu"][i] // 2),
        ]
        for i in range(len(class_ids))
    ]
    return table_of_images


def print_lane_locations_and_plot(rgbs, detector, class_ids):
    dashed = []
    solids = []
    for rgb_path in rgbs:
        seg = segmenter.segment(rgb_path)
        lanes = detect_class_lines(seg, detector, class_ids)
        dashed.append(lanes[1])
        solids.append(lanes[2])
        dashed_int = sorted([int(x) for x in lanes[1]])
        solids_int = sorted([int(x) for x in lanes[2]])
        print(f"dashed: {len(lanes[1])}, solid: {len(lanes[2])}", end="\t")
        print(f"d: {dashed_int}\ts:{solids_int}")

    for i in range(len(solids)):
        plt.scatter(solids[i], i * np.ones_like(solids[i]), c="green", s=1)
        plt.scatter(dashed[i], i * np.ones_like(dashed[i]), c="red", s=1)
    plt.xlim([-1500, 1500])
    plt.show()


def _scatter_plot(lines, ax):
    for k, v in lines.items():
        for i, l in enumerate(lines[k]):
            ax.scatter(l, i * np.ones_like(l), c=my_color_name(k), s=1)
    ax.set_xlim([-1500, 1500])


def detect_draw_img(rgb_path, detector, class_ids):
    rgb = imread(rgb_path)
    blank = np.zeros_like(rgb)
    blankc = np.zeros_like(rgb)
    shape, dtype = rgb.shape, rgb.dtype
    seg_path = segmenter.segment(rgb_path)
    seg = imread(seg_path)
    for cls in class_ids:
        seg_1class = apply_custom_map(seg, cls)
        can, _, _, long_centered_lines, xs = detector.detect(seg_1class, debug=True)
        blank += draw_lines(
            long_centered_lines, shape, dtype, VPOINT, my_color_rgb(cls)
        )
        if cls == 1:
            blankc += np.dstack([can, np.zeros_like(can), np.zeros_like(can)])
        elif cls == 2:
            blankc += np.dstack([np.zeros_like(can), can, np.zeros_like(can)])
        elif cls == 5:
            blankc += np.dstack([np.zeros_like(can), np.zeros_like(can), can])
        else:
            blankc += np.dstack([can, np.zeros_like(can), can])
    return rgb // 2 + blank // 2  #  + blankc //3


def test_visualize(rgb_paths_a, detector, rgb_paths_b, detector_b, class_ids):
    dataset = os.path.commonpath((rgb_paths_a[0], rgb_paths_b[0]))
    # folder = f"film_nobl_rh{detector.hough_rho}_the{detector.hough_theta}_thr{detector.hough_th}_ml{detector.hough_minlen}_mg{detector.hough_maxgap}_cth{detector.vpoint_th}_lth{detector.length_th}_{detector.cluster_th}"
    folder = f"mode2"
    folder = f"./{dataset}/" + folder

    try:
        shutil.rmtree(folder)
    except:
        pass
    os.mkdir(folder)
    lines_a = defaultdict(list)
    lines_b = defaultdict(list)
    data_a = []
    data_b = []

    for i, (rgb_path_a, rgb_path_b) in enumerate(zip(rgb_paths_a, rgb_paths_b)):
        img_a = detect_draw_img(rgb_path_a, detector, class_ids)
        seg = segmenter.segment(rgb_path_a)
        lanes_a = detect_class_lines(seg, detector, class_ids)
        for k, v in lanes_a.items():
            lines_a[k].append(v)
            for x in v:
                data_a.append([x, i, k])

        img_b = detect_draw_img(rgb_path_b, detector_b, class_ids)
        seg = segmenter.segment(rgb_path_b)
        lanes_b = detect_class_lines(seg, detector, class_ids)
        for k, v in lanes_b.items():
            lines_b[k].append(v)
            for x in v:
                data_b.append([x, i, k])

        fig = plt.figure(figsize=(16 * 2, 9 * 2))
        spec2 = gridspec.GridSpec(ncols=2, nrows=2, figure=fig, height_ratios=[1, 1])
        ax1 = fig.add_subplot(spec2[0, 0])
        ax2 = fig.add_subplot(spec2[0, 1])
        ax3 = fig.add_subplot(spec2[1, 0])
        ax4 = fig.add_subplot(spec2[1, 1])
        ax1.imshow(img_a)
        ax2.imshow(img_b)
        _scatter_plot(lines_a, ax3)
        ax3.set_ylim([-1, len(rgb_paths_a)])
        _scatter_plot(lines_b, ax4)
        ax4.set_ylim([-1, len(rgb_paths_b)])
        plt.tight_layout()
        # plt.show()
        plt.savefig(f"{folder}/film_{i:04}.png")
        plt.close()

    data_a = np.array(data_a)
    np.save(f"{folder}/data_a.npy", data_a)

    data_b = np.array(data_b)
    np.save(f"{folder}/data_b.npy", data_b)


def _mpl_show_data_simple(data, ax, xlim=None, ylim=0):
    ax.scatter(data, range(len(data)), c="white", s=1, marker="s")
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim != 0:
        n = int(np.max(data["frame"]))
        if n > ylim:
            ax.set_ylim([n - ylim, n])
        else:
            ax.set_ylim([0, ylim])


def _mpl_show_data(data, ax, true_ratio=False, xlim=0, ylim=0):
    for cls in set(data["class"]):
        datan = data[data["class"] == cls]
        ax.scatter(
            datan["location"], datan["frame"], c=my_color_name(cls), s=1, marker="s"
        )
    if xlim != 0:
        if np.isscalar(xlim):
            ax.set_xlim([-xlim, xlim])
        else:
            ax.set_xlim(xlim)
            plt.xticks([0, 1])
    if ylim != 0:
        n = int(np.max(data["frame"]))
        if n > ylim:
            ax.set_ylim([n - ylim, n])
        else:
            ax.set_ylim([0, ylim])
    if true_ratio:
        # ax.set_xlim([-5000, 5000])
        # ax.set_ylim([0, 10000])
        ax.set_aspect(1)


def _bokeh_show_data(data, fig):
    for cls in set(data["class"]):
        datan = data[data["class"] == cls]
        fig.square(datan["location"], datan["frame"], size=3, color=my_color_name(cls))


def mpl_show_data_twin(data, xlim=2000):
    fig = plt.figure(figsize=(16 * 1.5, 9 * 1.5))
    spec2 = gridspec.GridSpec(4, 6, figure=fig)

    ax0 = fig.add_subplot(spec2[:, 0:2])
    ax0.set_title("run 1")
    ax0.set_xlabel("detected lane location; color = class")
    ax0.set_ylabel("frame id")
    _mpl_show_data(data[0], ax0, xlim=xlim)

    ax1 = fig.add_subplot(spec2[:, 2:4])
    ax1.set_title("run 2")
    ax1.set_xlabel("detected lane location; color = class")
    _mpl_show_data(data[1], ax1, xlim=xlim)

    ax2 = fig.add_subplot(spec2[:, 4])
    ax2.set_xlabel("#consequtive detected differences")
    _mpl_show_data_simple(data[2], ax2)

    ax3 = fig.add_subplot(spec2[:, 5])
    ax3.set_xlabel("change detection")
    _mpl_show_data_simple(data[3], ax3, xlim=[-0.25, 1.25])

    plt.tight_layout()
    # plt.show()
    return fig


def mpl_show_data_twin_plus_imgs(img_a, img_b, data, xlim=2000, ylim=100, i=0):
    fig = plt.figure(figsize=(16 * 1.2, 9 * 1.2))
    spec2 = gridspec.GridSpec(6, 6, figure=fig)

    axa = fig.add_subplot(spec2[:2, 0:2])
    axa.set_title(f"run 1: frame {i}")
    axa.imshow(img_a)

    axb = fig.add_subplot(spec2[:2, 2:4])
    axb.set_title(f"run 2: frame {i}")
    axb.imshow(img_b)
    axb = fig.add_subplot(spec2[:2, 2:4])

    ax0 = fig.add_subplot(spec2[2:, 0:2])
    ax0.set_title("run 1")
    ax0.set_xlabel("detected lane location; color = class")
    ax0.set_ylabel("frame id")
    _mpl_show_data(data[0], ax0, xlim=xlim, ylim=ylim)

    ax1 = fig.add_subplot(spec2[2:, 2:4])
    ax1.set_title("run 2")
    ax1.set_xlabel("detected lane location; color = class")
    _mpl_show_data(data[1], ax1, xlim=xlim, ylim=ylim)

    ax2 = fig.add_subplot(spec2[2:, 4])
    ax2.set_xlabel("#consequtive detected differences")
    _mpl_show_data(data[2], ax2, ylim=ylim)

    ax3 = fig.add_subplot(spec2[2:, 5])
    ax3.set_xlabel("change detection")
    _mpl_show_data(data[3], ax3, xlim=[-0.25, 1.25], ylim=ylim)

    plt.tight_layout()
    # plt.show()
    return fig


def bokeh_show_images(img1, img2):
    i1 = bokeh.plotting.figure(
        plot_width=800,
        plot_height=400,
        title=None,
        x_range=(0, 1920),
        y_range=(-1080, 0),
    )
    i1.image_url(url=[img1], x=0, y=0, w=1920, h=1080)
    i2 = bokeh.plotting.figure(
        plot_width=800,
        plot_height=400,
        title=None,
        x_range=(0, 1920),
        y_range=(-1080, 0),
    )
    i2.image_url(url=[img2], x=0, y=0, w=1920, h=1080)
    return bokeh.layouts.row(i1, i2)


def bokeh_show_data_twin(data, xlim=2500):
    """data - list of"""
    n = len(data)
    assert n >= 2
    figs = [
        bokeh.plotting.figure(
            plot_width=400,
            plot_height=600,
            x_range=(-xlim, xlim),
            title=None,
            tools="pan,ywheel_zoom,box_zoom,reset",
            active_scroll="ywheel_zoom",
        )
    ]
    _bokeh_show_data(data[0], figs[0])
    figs.append(
        bokeh.plotting.figure(
            plot_width=400,
            plot_height=600,
            x_range=(-xlim, xlim),
            y_range=figs[0].y_range,
            title=None,
            tools="pan,ywheel_zoom,box_zoom,reset",
            active_scroll="ywheel_zoom",
        )
    )
    _bokeh_show_data(data[1], figs[1])
    for i in range(2, n):
        figs.append(
            bokeh.plotting.figure(
                plot_width=400, plot_height=600, y_range=figs[0].y_range, title=None
            )
        )
        figs[-1].square(data[i], range(len(data[i])), size=3)
    return bokeh.layouts.row(*figs)


def bokeh_show_data_twin_show(img1_path, img2_path, data, i):
    imgs = bokeh_show_images(img1_path, img2_path)
    plots = bokeh_show_data_twin(data)
    full = bokeh.layouts.layout([[imgs], [plots]])
    bokeh.io.show(full)
    # bokeh.io.export_png(full, filename=f"p{i}.png")


def _get_filename_int(path: str):
    base = os.path.basename(path)
    filename, ext = os.path.splitext(base)
    return int(filename)


def get_images(dataset, M=0, N=0):
    rgb_paths_a = glob(f"./{dataset}/run1/*.jpg")
    rgb_paths_a.sort()
    rgb_paths_b = glob(f"./{dataset}/run2/*.jpg")
    rgb_paths_b.sort()
    if M != 0 or N != 0:
        rgb_paths_a = rgb_paths_a[M:N]
        rgb_paths_b = rgb_paths_b[M:N]
    return rgb_paths_a, rgb_paths_b


### MAIN ###


def main():
    args = get_args()
    rgb_paths_a, rgb_paths_b = get_images(args.dataset)

    detector = LaneDetector(VPOINT)
    detector_b = LaneDetector(VPOINT_B)

    class_ids = [1, 2, 5]  # , 7, 9]

    if args.mode == "mode0":
        for rgb_path_a, rgb_path_b in zip(rgb_paths_a, rgb_paths_b):
            ta = visualize_detection_stages(rgb_path_a, detector, class_ids)
            tb = visualize_detection_stages(rgb_path_b, detector_b, class_ids)
            table_of_images = ta + tb
            # stack_imshow(*table_of_images)
            gridspec_imshow(*table_of_images)
            plt.savefig(f"./{args.dataset}/debug_a.png")

    if args.mode == "mode1":
        print_lane_locations_and_plot(rgb_paths_a, detector, class_ids)

    if args.mode == "mode2":
        test_visualize(rgb_paths_a, detector, rgb_paths_b, detector_b, class_ids)

    if args.mode == "mode3" or args.mode == "mode4":
        if not os.path.exists(f"./{args.dataset}/data_a.npy"):
            data = detect_fast(rgb_paths_a, segmenter, detector, class_ids)
            np.save(f"./{args.dataset}/data_a.npy", data)
            print("Saved data_a.npy")
        if not os.path.exists(f"./{args.dataset}/data_b.npy"):
            data = detect_fast(rgb_paths_b, segmenter, detector_b, class_ids)
            np.save(f"./{args.dataset}/data_b.npy", data)
            print("Saved data_b.npy")
        data_a = np.load(f"./{args.dataset}/data_a.npy")
        data_b = np.load(f"./{args.dataset}/data_b.npy")

    if args.mode == "mode3":
        data_ap = process(data_a)
        data_bp = process(data_b)
        data_diff = compare(data_ap, data_bp)
        data_gated = gate(data_diff, args.diff_threshold)
        f = mpl_show_data_twin([data_ap, data_bp, data_diff, data_gated])
        plt.savefig(f"./{args.dataset}/full.png")
        plots = bokeh_show_data_twin([data_ap, data_bp, data_diff, data_gated])
        bokeh.io.show(plots)

    elif args.mode == "mode4":
        for i in range(len(rgb_paths_a)):
            img_a = detect_draw_img(rgb_paths_a[i], detector, class_ids)
            img_b = detect_draw_img(rgb_paths_b[i], detector_b, class_ids)
            data_an = data_a[data_a[:, 1] <= i]
            data_bn = data_b[data_b[:, 1] <= i]
            data_ap = process(data_an)
            data_bp = process(data_bn)
            data_diff = compare(data_ap, data_bp)
            data_gated = gate(data_diff)

            # bokeh has issues with saving the images of plots with images in them
            # bokeh_show_data_twin_show( rgb_paths_a[i], rgb_paths_b[i], [data_ap, data_bp, data_diff, data_gated], i)

            f = mpl_show_data_twin_plus_imgs(
                img_a,
                img_b,
                [data_ap, data_bp, data_diff, data_gated],
                ylim=args.ylimit,
                i=i,
            )
            plt.savefig(f"./{args.dataset}/film/{i:04d}.png")


if __name__ == "__main__":
    t0 = time()
    main()
    print("Time:", timedelta(seconds=time() - t0))
