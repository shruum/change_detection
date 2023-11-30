import glob
import json
import numpy as np
from PIL import Image, ImageDraw
import sys


def get_files(folder_name):
    read_files = glob.glob(folder_name + "/*.jpg")
    files = []
    for file in read_files:
        files.append(file)

    return files


def draw_polygons_from_json(json_file, img_files, output_folder):
    with open(json_file) as file:
        imgs = json.load(file)["imgs"]

        for f in img_files:
            file_name_with_path = f.split("/")[-1]
            file_name = file_name_with_path.split(".")[0]
            print("Color polygons for " + file_name + " ...")
            draw_polygons(f, imgs[file_name]["objects"], file_name, output_folder)


def draw_polygons(file, objects, file_name, output_folder):
    im = Image.open(file)
    im_pred = Image.new(mode="RGB", size=im.size)
    draw = ImageDraw.Draw(im_pred)

    for obj in objects:
        poly = obj["polygon"]
        vertices = np.asarray(poly)
        vertices = vertices.reshape(len(vertices) // 2, 2)
        col = tuple(obj["color"])
        if len(vertices) == 1:
            draw.point([(x, y) for x, y in vertices], fill=col)
        else:
            draw.polygon([(x, y) for x, y in vertices], fill=col, outline=None)

    # im_pred.show()
    im_pred.save(output_folder + "/" + file_name + ".png")


if __name__ == "__main__":
    json_file = sys.argv[1]
    data_folder = sys.argv[2]
    output_folder = sys.argv[3]

    img_files = get_files(data_folder)
    draw_polygons_from_json(json_file, img_files, output_folder)
