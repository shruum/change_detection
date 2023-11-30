import cv2
import glob
import json
import numpy as np
from PIL import Image
import sys


def get_files(folder_name):
    read_files = glob.glob(folder_name + "/*.jpg")
    files = []
    for file in read_files:
        files.append(file)

    return files


def get_labels(config_file):
    with open(config_file) as config:
        config = json.load(config)

    return config["labels"]


def apply_color_map(image_array, labels):
    color_array = np.zeros(
        (image_array.shape[0], image_array.shape[1], 3), dtype=np.uint8
    )

    for label_id, label in enumerate(labels):
        # set all pixels with the current label to the color of the current label
        color_array[image_array == label_id] = label["color"]

    color_array = Image.fromarray(color_array)

    return color_array


def color_predicts(files, predicts_folder, output_folder, labels):
    for file in files:
        img = cv2.imread(file)
        img_file_name = file.split("/")[-1]
        file_name = img_file_name.split(".")[0] + ".dat"
        print("Color " + img_file_name + " prediction ...")

        predict_cpp = np.fromfile(
            predicts_folder + "/" + file_name,
            dtype=np.byte,
            count=(img.shape[0] * img.shape[1]),
        )
        predict_cpp = np.reshape(predict_cpp, (img.shape[0], img.shape[1]))
        predict_cpp_colored = apply_color_map(predict_cpp, labels)
        predict_cpp_colored.save(output_folder + "/" + file_name.split(".")[0] + ".png")


if __name__ == "__main__":
    data_folder = sys.argv[1]
    predicts_folder = sys.argv[2]
    output_folder = sys.argv[3]
    config_file = sys.argv[4]

    files = get_files(data_folder)
    labels = get_labels(config_file)

    color_predicts(files, predicts_folder, output_folder, labels)
