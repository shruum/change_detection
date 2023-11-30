import argparse
import glob
import json
import os
from os import path
import pandas as pd
from PIL import Image
import numpy as np
import shutil
from vizer.draw import _draw_single_box


def mkdir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


def overlay_det_results(frame_id, tracks, detections, output_path, input_data):

    output_path = os.path.join(output_path, "overlay")
    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    image_path = input_data[frame_id]
    output_path = os.path.join(output_path, os.path.basename(image_path))
    if os.path.exists(output_path):
        image = np.array(Image.open(output_path).convert("RGB"))
    else:
        image = np.array(Image.open(image_path).convert("RGB"))

    if len(detections) > 0:
        for det in detections:
            display_string = "Det:" + "{:.5s}".format(str(det[5]))
            drawn_image = _draw_single_box(
                Image.fromarray(image),
                det[1],
                det[0],
                det[3],
                det[2],
                (255, 0, 0),
                str(display_string),
            )
            drawn_image = np.array(drawn_image, dtype=np.uint8)
            Image.fromarray(drawn_image).save(output_path)
    else:
        Image.fromarray(image).save(output_path)


def overlay_track_results(frame_id, tracks, detections, output_path, input_data):

    output_path = os.path.join(output_path, "overlay")
    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    for track in tracks:
        if len(track.bbox_list) > 0:
            for index, frame_id in enumerate(track.frame_ids):
                image_path = input_data[frame_id]
                output_img_path = os.path.join(
                    output_path, os.path.basename(image_path)
                )
                if os.path.exists(output_img_path):
                    image = np.array(Image.open(output_img_path).convert("RGB"))
                else:
                    image = np.array(Image.open(image_path).convert("RGB"))

                bbox_list = track.bbox_list[index]
                metadata = track.metadata[0]

                display_string = (
                    "C: "
                    + "{:.5s}".format(str(metadata[-1]))
                    # + "\n"
                    # + "D: "
                    # + "{:.5s}".format(str(metadata[1]))
                )
                drawn_image = _draw_single_box(
                    Image.fromarray(image),
                    bbox_list[1],
                    bbox_list[0],
                    bbox_list[3],
                    bbox_list[2],
                    (0, 255, 0),
                    str(display_string),
                )
                drawn_image = np.array(drawn_image, dtype=np.uint8)
                Image.fromarray(drawn_image).save(output_img_path)


def overlay_detections(detect_sequence, input_data, output_dir):

    for i, dt in enumerate(detect_sequence):
        image_path = input_data[i]
        output_path = os.path.join(output_dir, os.path.basename(image_path))
        if path.exists(output_path):
            image = np.array(Image.open(output_path).convert("RGB"))
        else:
            image = np.array(Image.open(image_path).convert("RGB"))
        with open(dt) as json_file:
            data = json.load(json_file)
            for obj in data["objects"]:
                det_class = obj["f_name"]
                for j in range(len(obj["obj_points"])):
                    x1 = obj["obj_points"][j]["x"]
                    y1 = obj["obj_points"][j]["y"]
                    x2 = obj["obj_points"][j]["x"] + obj["obj_points"][j]["w"]
                    y2 = obj["obj_points"][j]["y"] + obj["obj_points"][j]["h"]
                    det_score = obj["obj_points"][j]["f_conf"]
                    display_string = (
                        "Det : " + det_class + "\t" + "Score : " + str(det_score)
                    )
                    drawn_image = _draw_single_box(
                        Image.fromarray(image),
                        int(x1),
                        int(y1),
                        int(x2),
                        int(y2),
                        (255, 0, 0),
                        str(display_string),
                    )
                    drawn_image = np.array(drawn_image, dtype=np.uint8)
                    # image_name, ext = os.path.splitext(os.path.basename(image_path))
                    Image.fromarray(drawn_image).save(
                        os.path.join(output_dir, os.path.basename(image_path))
                    )


def overlay_tracks(track_sequence, input_data, output_dir):

    for tr in range(len(track_sequence)):
        track = track_sequence[tr]
        dirname = os.path.basename(os.path.dirname((track)))
        track_file = track + "bbox_list" + str(dirname)
        if not path.exists(track_file):
            print(track_file)
            continue
        file = pd.read_csv(track_file, sep=";")
        box = np.ones((len(file), 4))
        box[:, 0] = np.asarray(file.x0)
        box[:, 1] = np.asarray(file.y0)
        box[:, 2] = np.asarray(file.x1)
        box[:, 3] = np.asarray(file.y1)
        frames = np.asarray(file.frame_id)
        det_class = np.asarray(file.label_id)
        classfn_class = np.asarray(file.conf_class)
        conf_detec = np.asarray(file.conf_detec)
        conf_detec = conf_detec[0]
        detect_score = np.asarray(file.detect_score)
        conf_score = np.asarray(file.conf_detec)

        if conf_detec > 0.7:
            for fr in range(len(frames)):
                image_path = input_data[frames[fr]]
                output_path = os.path.join(output_dir, os.path.basename(image_path))
                if path.exists(output_path):
                    image = np.array(Image.open(output_path).convert("RGB"))
                else:
                    image = np.array(Image.open(image_path).convert("RGB"))

                display_string = (
                    "Classfn : "
                    + classfn_class[0]
                    + "\t"
                    + "Score : "
                    + "{:.5s}".format(str(conf_score[fr]))
                )

                # drawn_image = draw_boxes(image, box[fr,:], labels).astype(np.uint8)
                drawn_image = _draw_single_box(
                    Image.fromarray(image),
                    box[fr, 0],
                    box[fr, 1],
                    box[fr, 2],
                    box[fr, 3],
                    (0, 255, 0),
                    str(display_string),
                )
                drawn_image = np.array(drawn_image, dtype=np.uint8)
                # image_name, ext = os.path.splitext(os.path.basename(image_path))
                Image.fromarray(drawn_image).save(
                    os.path.join(output_dir, os.path.basename(image_path))
                )


def overlay(args):

    input_file_list = glob.glob(os.path.join(args.images_dir, "*", ""))
    track_file_list = glob.glob(os.path.join(args.tracks_dir, "*", ""))
    detect_file_list = glob.glob(os.path.join(args.detections_dir, "*", ""))

    for sequence_id in range(len(input_file_list)):  # Loop through each sequence
        output_dir = os.path.join(args.output_dir, str(sequence_id + 1))
        mkdir(output_dir)

        input_data = []
        input_data_path = input_file_list[sequence_id]
        input_data += glob.glob(os.path.join(input_data_path, "*.png"))
        input_data += glob.glob(os.path.join(input_data_path, "*.jpg"))
        input_data.sort()

        track_path = track_file_list[sequence_id]
        track_sequence = glob.glob(os.path.join(track_path, "*", ""))
        track_sequence.sort()

        if args.overlay_detections:
            detect_path = detect_file_list[sequence_id]
            detect_sequence = glob.glob(os.path.join(detect_path, "*.json"))
            detect_sequence.sort()
            overlay_detections(detect_sequence, input_data, output_dir)

        overlay_tracks(track_sequence, input_data, output_dir)


def main():
    parser = argparse.ArgumentParser(description="Tracker Demo.")
    parser.add_argument(
        "--images_dir", type=str, help="Specify a image dir to do prediction."
    )
    parser.add_argument(
        "--tracks_dir", type=str, help="Specify a image dir to do prediction."
    )
    parser.add_argument(
        "--output_dir",
        default="demo/result",
        type=str,
        help="Specify a image dir to save predicted images.",
    )
    parser.add_argument(
        "--detections_dir", type=str, help="Specify detection files dir."
    )
    parser.add_argument(
        "--overlay_detections",
        action="store_true",
        help="To overlay detections seperately from tracks",
    )

    args = parser.parse_args()
    overlay(args)


if __name__ == "__main__":
    main()
