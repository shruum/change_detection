from collections import OrderedDict
import glob
import json
import glob
import logging
import mimetypes
import numpy as np
from shutil import copy2
from tqdm import tqdm
import xmltodict
from xml.dom.minidom import parseString
import xml.etree.cElementTree as ET

mimetypes.init()
import os
import sys

CURRENT_DIR = os.path.dirname(__file__)
sys.path.insert(0, CURRENT_DIR)


def xywh_to_xyxy(bbox):
    x0, y0, w, h = bbox
    x1 = x0 + w - 1
    y1 = y0 + h - 1
    return [x0, y0, x1, y1]


def xyxy_to_xywh(bbox):
    x0, y0, x1, y1 = bbox
    w = x1 - x0 + 1
    h = y1 - y0 + 1
    return [x0, y0, w, h]


def check_bbox(bbox):
    try:
        bbox = [int(x) for x in bbox]
        return bbox
    except:
        return None


def find_match(id, match_array):
    index = np.nonzero(id == np.array(match_array))[0]
    if len(index) > 0:
        return index[0]
    else:
        return None


def find_match_else_add(id, match_array):
    index = find_match(id, match_array)
    if index is None:
        index = len(match_array)
        match_array.append(id)
    return index, match_array


def fix_xml_object(annotation_dict):
    if isinstance(annotation_dict["object"], OrderedDict):
        annotation_dict["object"] = [annotation_dict["object"]]
    return annotation_dict


def get_detections_from_json_dict(json_dict):
    detection_list = []
    if "objects" in json_dict:
        detection_object_list = json_dict["objects"]
        for detection_object in detection_object_list:
            if "obj_points" in detection_object:
                class_id = detection_object["f_code"]
                class_name = detection_object.get("f_name")
                confidence = detection_object.get("f_conf")
                bbox = detection_object["obj_points"]
                detection = [
                    bbox["y"],
                    bbox["x"],
                    bbox["h"],
                    bbox["w"],
                    class_name,
                    class_id,
                    confidence,
                ]
            elif "bbox" in detection_object:
                class_name = detection_object["category"]
                bbox = detection_object["bbox"]
                confidence = detection_object["score"] / 100.0
                x0, y0, w, h = xyxy_to_xywh(
                    [bbox["xmin"], bbox["ymin"], bbox["xmax"], bbox["ymax"]]
                )
                detection = [
                    y0,
                    x0,
                    h,
                    w,
                    class_name,
                    None,
                    confidence,
                ]
            detection_list.append(detection)
    return detection_list


def get_detections_from_xml_dict(annotation_dict):
    detection_list = []
    if "object" in annotation_dict:
        annotation_dict = fix_xml_object(annotation_dict)
        for annotation_object in annotation_dict["object"]:
            x0, y0, x1, y1, class_name = get_bbox_and_class_from_xml_object(
                annotation_object
            )
            x0, y0, w, h = xyxy_to_xywh([x0, y0, x1, y1])
            if "confidence" in annotation_object:
                confidence = float(annotation_object["confidence"])
            else:
                confidence = None
            detection_list.append([y0, x0, h, w, class_name, None, confidence])
    return detection_list


def get_bbox_and_class_from_xml_object(annotation_object):
    class_name = annotation_object["name"]
    bbox = annotation_object["bndbox"]
    x0 = int(float(bbox["xmin"]))
    y0 = int(float(bbox["ymin"]))
    x1 = int(float(bbox["xmax"]))
    y1 = int(float(bbox["ymax"]))
    return x0, y0, x1, y1, class_name


def load_xml(xml_path):
    return xmltodict.parse(open(xml_path).read())


def load_xml_annotation_dict(xml_path):
    return load_xml(xml_path)["annotation"]


def save_xml(xml_result, output_path):
    with open(output_path, mode="w") as file:
        file.write(xml_result)


def xml_dict_to_string(xml_dict):
    xml_result = xmltodict.unparse(xml_dict)
    dom = parseString(xml_result)
    return dom.toprettyxml()


def load_json(json_path):
    with open(json_path) as json_file:
        return json.load(json_file)


def save_json(json_dict, output_path):
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(json_dict, file, ensure_ascii=False, indent=3)


def convert_to_xml(detections, img_filename, img_width, img_height, foldername=[]):
    # Input:
    #   detections: [y, x, h, w, class_name, class_id, confidence]
    node_root = ET.Element("annotation")
    node_folder = ET.SubElement(node_root, "folder")
    if not foldername:
        node_folder.text = os.path.dirname(os.path.abspath(img_filename))
    else:
        node_folder.text = foldername
    node_filename = ET.SubElement(node_root, "filename")
    node_filename.text = os.path.basename(img_filename)
    node_size = ET.SubElement(node_root, "size")
    ET.SubElement(node_size, "width").text = str(img_width)
    ET.SubElement(node_size, "height").text = str(img_height)
    for detection in detections:
        y, x, h, w, class_name, class_id, confidence = detection
        node_object = ET.SubElement(node_root, "object")
        ET.SubElement(node_object, "name").text = str(class_name)
        ET.SubElement(node_object, "difficult").text = "0"
        object_bndbox = ET.SubElement(node_object, "bndbox")
        ET.SubElement(object_bndbox, "xmin").text = str(int(x))
        ET.SubElement(object_bndbox, "ymin").text = str(int(y))
        ET.SubElement(object_bndbox, "xmax").text = str(int(x + w - 1))
        ET.SubElement(object_bndbox, "ymax").text = str(int(y + h - 1))
        ET.SubElement(node_object, "confidence").text = str(confidence)
    detection_tree = ET.tostring(node_root)
    dom = parseString(detection_tree)
    return dom.toprettyxml()


def nie_json_to_xml(json_path, output_path, foldername=".", extension=".jpg"):
    json_dict = load_json(json_path)
    img_width = json_dict["width"] if "width" in json_dict else None
    img_height = json_dict["height"] if "height" in json_dict else None
    filename = get_basename_no_ext(json_path) + extension
    detections = get_detections_from_json_dict(json_dict)
    xml_result = convert_to_xml(
        detections, filename, img_width, img_height, foldername=foldername
    )
    save_xml(xml_result, output_path)


def nie_json_to_xml_dir(json_dir, output_dir, foldername=".", extension=".jpg"):
    os.makedirs(output_dir, exist_ok=True)
    json_path_list = sorted(glob.glob(os.path.join(json_dir, "*.json")))
    for json_index in tqdm(range(len(json_path_list))):
        json_path = json_path_list[json_index]
        basename = get_basename_no_ext(json_path)
        output_path = os.path.join(output_dir, basename + ".xml")
        nie_json_to_xml(
            json_path, output_path, foldername=foldername, extension=extension
        )


def xml_to_nie_json(xml_path):
    doc = load_xml(xml_path)
    annotation_dict = doc["annotation"]
    img_width = annotation_dict["size"]["width"]
    img_height = annotation_dict["size"]["height"]
    detection_list = get_detections_from_xml_dict(annotation_dict)
    result = create_nie_json(img_width, img_height, detection_list)
    result["filename"] = annotation_dict["filename"]
    return result


def xml_to_nie_json_dir(xml_dir, output_dir):
    xml_path_list = sorted(glob.glob(os.path.join(xml_dir, "*.xml")))
    for xml_index in tqdm(range(len(xml_path_list))):
        xml_path = xml_path_list[xml_index]
        basename = get_basename_no_ext(xml_path)
        output_path = os.path.join(output_dir, basename + ".json")
        json_dict = xml_to_nie_json(xml_path)
        save_json(json_dict, output_path)


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def get_detections_from_file(filepath):
    ext = os.path.splitext(filepath)[1]
    if ext == ".xml":
        annotation_dict = load_xml_annotation_dict(filepath)
        detection_list = get_detections_from_xml_dict(annotation_dict)
    elif ext == ".json":
        annotation_dict = load_json(filepath)
        detection_list = get_detections_from_json_dict(annotation_dict)
    return detection_list


def get_annotation_dict(filepath):
    ext = os.path.splitext(filepath)[1]
    if ext == ".xml":
        annotation_dict = load_xml_annotation_dict(filepath)
    elif ext == ".json":
        annotation_dict = load_json(filepath)
    return annotation_dict


def calculate_intersection(box1, box2):
    if box1[3] < box2[1] or box1[1] > box2[3] or box1[2] < box2[0] or box1[0] > box2[2]:
        return 0.0, 0.0, 0.0
    ixmin = max(box1[1], box2[1])
    iymin = max(box1[0], box2[0])
    ixmax = min(box1[3], box2[3])
    iymax = min(box1[2], box2[2])
    bbox_area_1 = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    bbox_area_2 = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    iw = ixmax - ixmin + 1
    ih = iymax - iymin + 1
    ia = float(iw * ih)
    return ia, bbox_area_1, bbox_area_2


def calculate_iou(box1, box2):
    intersection, bbox_area_1, bbox_area_2 = calculate_intersection(box1, box2)
    union = bbox_area_1 + bbox_area_2 - intersection
    if union > 0.0:
        return float(intersection) / float(union)
    else:
        return 0.0


def get_image_list(data_dir, recursive=True):
    extension_list = get_extensions_for_type("image")
    image_path_list = []
    for extension in extension_list:
        if recursive:
            image_path_list += glob.glob(
                os.path.join(data_dir, "**", "*" + extension), recursive=True
            )
        else:
            image_path_list += glob.glob(os.path.join(data_dir, "*" + extension))
    return image_path_list


def get_extensions_for_type(general_type):
    for ext in mimetypes.types_map:
        if mimetypes.types_map[ext].split(os.sep)[0] == general_type:
            yield ext


def find_img_extension(image_basename):
    extension_list = get_extensions_for_type("image")
    for ext in extension_list:
        image_path = image_basename + ext
        if os.path.isfile(image_path):
            return image_path
    return None


def get_basename_no_ext(filepath):
    return os.path.splitext(os.path.basename(filepath))[0]


def parse_txt_annotation_file(
    annotation_filepath, image_dir, img_width=None, img_height=None, delimiter=";"
):
    from ni_standardization.converter import converter

    Converter = converter()
    image_path = os.path.join(image_dir, get_basename_no_ext(annotation_filepath))
    image_path = find_img_extension(image_path)
    if image_path is None:
        logging.warning(
            "No associated image found for annotation file:", annotation_filepath
        )
    with open(annotation_filepath) as file:
        annotations = file.readlines()
    if img_width is None or img_height is None:
        img = Converter._read_image(image_path)
        img_width, img_height = img.size
    for line in annotations:
        yield Converter._parse_annotation_line(
            line, img_width=img_width, img_height=img_height, delimiter=delimiter
        )


def create_nie_json(img_width, img_height, detections):
    # Input:
    #   detections: [[y, x, h, w, class_name, class_id, confidence, instance_id], [...], [...]]
    # Save output
    output_dict = dict()
    output_dict["width"] = int(img_width)
    output_dict["height"] = int(img_height)
    output_dict["objects"] = []
    for detection in detections:
        object_dict = dict()
        bbox = detection[:4]
        object_dict["f_name"] = detection[4]
        object_dict["f_code"] = detection[5]
        object_dict["obj_points"] = {}
        object_dict["obj_points"]["x"] = int(bbox[1])
        object_dict["obj_points"]["y"] = int(bbox[0])
        object_dict["obj_points"]["w"] = int(bbox[3])
        object_dict["obj_points"]["h"] = int(bbox[2])
        object_dict["f_conf"] = detection[6] if len(detection) >= 7 else None
        object_dict["instance_id"] = detection[7] if len(detection) >= 8 else None
        output_dict["objects"].append(object_dict)
    output_dict["obj_num"] = len(output_dict["objects"])
    return output_dict


def save_nie_json(img_width, img_height, detections, output_path):
    output_dict = create_nie_json(img_width, img_height, detections)
    save_json(output_dict, output_path)


def convert_coco_to_json_nie(annotation_contents):
    annotation_dict = {}
    annotation_list = annotation_contents["annotations"]
    for annotation in annotation_list:
        image_id = str(annotation["image_id"])
        if image_id not in annotation_dict:
            annotation_dict[image_id] = {}
            annotation_dict[image_id]["annotations"] = []
        annotation_dict[image_id]["annotations"].append(annotation)
    image_list = annotation_contents["images"]
    for image_entry in image_list:
        image_id = str(image_entry["id"])
        for key in image_entry:
            if image_id not in annotation_dict:
                annotation_dict[image_id] = {}
            annotation_dict[image_id][key] = image_entry[key]
    return annotation_dict


def copy_images_from_coco_json(json_path, image_dir, output_dir):
    json_dict = load_json(json_path)
    image_dict = json_dict["images"]
    for entry in image_dict:
        image_path = os.path.join(image_dir, entry["file_name"])
        copy2(image_path, output_dir)


def non_maximum_surpression(
    bbox_list_1, bbox_list_2, iou_threshold=0.9, auto_nsm=False
):
    # bbox: [x0, y0, x1, y1, score]
    to_remove_1 = np.zeros(len(bbox_list_1)).astype(bool)
    to_remove_2 = np.zeros(len(bbox_list_2)).astype(bool)
    for bbox_index_1, bbox_1 in enumerate(bbox_list_1):
        for bbox_index_2, bbox_2 in enumerate(bbox_list_2):
            if auto_nsm and bbox_index_1 == bbox_index_2:
                continue
            iou = calculate_iou(bbox_1, bbox_2)
            if iou > iou_threshold:
                score_1 = bbox_1[4]
                score_2 = bbox_2[4]
                if score_1 > score_2:
                    to_remove_2[bbox_index_2] = True
                else:
                    to_remove_1[bbox_index_1] = True
    return to_remove_1, to_remove_2


def non_maximum_surpression_file(filepath, output_dir):
    annotation_dict = get_annotation_dict(filepath)
    img_width = annotation_dict["size"]["width"]
    img_height = annotation_dict["size"]["height"]
    detection_list = get_detections_from_file(filepath)
    bbox_list = []
    for detection in detection_list:
        y0, x0, h, w, _, _, confidence = detection
        x0, y0, x1, y1 = xywh_to_xyxy([x0, y0, w, h])
        bbox_list.append([x0, y0, x1, y1, confidence])
    to_remove, _ = non_maximum_surpression(bbox_list, bbox_list, auto_nsm=True)
    detection_list = list(np.array(detection_list)[~to_remove])
    xml_result = convert_to_xml(
        detection_list, annotation_dict["filename"], img_width, img_height
    )
    output_path = os.path.join(output_dir, os.path.basename(filepath))
    save_xml(xml_result, output_path)


def convert_track_json_to_nie_json(track_json_filepath, output_dir):
    # Convert output from `nie_tracking/visual_tracking`
    os.makedirs(output_dir, exist_ok=True)
    track_json_dict = load_json(track_json_filepath)
    instance_id = track_json_dict["instance_id"]
    img_width = track_json_dict["width"]
    img_height = track_json_dict["height"]
    for object_dict in track_json_dict["objects"]:
        frame_id = object_dict["frame_id"]
        x0 = object_dict["x0"]
        y0 = object_dict["y0"]
        x1 = object_dict["x1"]
        y1 = object_dict["y1"]
        w = x1 - x0
        h = y1 - y0
        class_name = object_dict["detection_class"]
        confidence = object_dict["detection_score"]
        output_path = os.path.join(output_dir, str(frame_id).zfill(6) + ".json")
        detections = []
        detections.append([y0, x0, h, w, class_name, None, confidence, instance_id])
        if os.path.isfile(output_path):  # Append existing annotations
            for detection in get_detections_from_file(output_path):
                detections.append(detection)
        save_nie_json(img_width, img_height, detections, output_path)


def convert_track_json_to_nie_json_dir(track_dir, output_dir):
    for track_json_filepath in glob.glob(os.path.join(track_dir, "*.json")):
        convert_track_json_to_nie_json(track_json_filepath, output_dir)
