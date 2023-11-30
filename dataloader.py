# Copyright (C) 2020 by NavInfo Europe B.V. The Netherlands - All rights reserved
# Information classification: Confidential
# This content is protected by international copyright laws.
# Reproduction and distribution is prohibited without written permission.
import csv
from geopy.distance import vincenty
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pathlib
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm


def list_to_string(s):
    str1 = ""
    for val in s:
        str1 = os.path.join(str1, val)
    return str1


def write_images(aligned_array, aligned_cords, output_dir):
    save_dir1 = os.path.join(output_dir, "aligned_images/run1")
    save_dir2 = os.path.join(output_dir, "aligned_images/run2")
    os.makedirs(save_dir1, exist_ok=True)
    os.makedirs(save_dir2, exist_ok=True)
    total = aligned_array.shape[0]
    for n, name_pair in enumerate(aligned_array):
        img1 = cv2.imread(name_pair[0])
        img2 = cv2.imread(name_pair[1])
        try:
            if aligned_cords is not None:
                coord1 = aligned_cords[n, 0]
                coord2 = aligned_cords[n, 1]
                cv2.imwrite(
                    os.path.join(
                        save_dir1,
                        "img_{:06d}_{}_{}.jpg".format(n, coord1[0], coord1[1]),
                    ),
                    img1,
                )
                cv2.imwrite(
                    os.path.join(
                        save_dir2,
                        "img_{:06d}_{}_{}.jpg".format(n, coord2[0], coord2[1]),
                    ),
                    img2,
                )
                # cv2.imwrite(save_dir2 + "img_{:06d}_{}_{}.jpg".format(n, coord2[0], coord2[1]), img2)
            else:
                # coord1 = name_pair[0].split("_")[-2:]
                coord1 = os.path.splitext(name_pair[0])[0].split("_")[-2:]
                # coord2 = name_pair[1].split("_")[-2:]
                coord2 = os.path.splitext(name_pair[1])[0].split("_")[-2:]
                cv2.imwrite(
                    os.path.join(
                        save_dir1,
                        "img_{:06d}_{}_{}.jpg".format(n, coord1[0], coord1[1]),
                    ),
                    img1,
                )
                cv2.imwrite(
                    os.path.join(
                        save_dir2,
                        "img_{:06d}_{}_{}.jpg".format(n, coord2[0], coord2[1]),
                    ),
                    img2,
                )
                # cv2.imwrite(save_dir1 + "img_{:06d}_{}_{}".format(n, coord1[0], coord1[1]), img1)
                # cv2.imwrite(save_dir2 + "img_{:06d}_{}_{}".format(n, coord2[0], coord2[1]), img2)
        except:
            print(
                "Found unsuitable (corrupted/empty) in name_pair -> {}, skipping...".format(
                    name_pair
                )
            )
        print("{}/{}".format(n, total))
    print("Images saved at {}".format(output_dir))


class Loader(Dataset):
    def __init__(
        self,
        l_dataDir=None,
        r_dataDir=None,
        augmentation=False,
        height=512,
        width=1024,
        format="aiim",
        txt_info_1=None,
        txt_info_2=None,
        no_alignment_speed_correction=False,
    ):
        super(Loader, self).__init__()
        if not no_alignment_speed_correction:
            logging.info(
                "With speed correction the alignment lists will take time \nPlease wait..."
            )

        if format == "aiim" or format == "fangzhou":
            if format == "aiim":
                info_1 = read_text_file_aiim(txt_info_1)
                info_2 = read_text_file_aiim(txt_info_2)
            elif format == "fangzhou":
                info_1 = read_text_file_fangzhou(txt_info_1)
                info_2 = read_text_file_fangzhou(txt_info_2)
            info_1 = self.interpolate_repeated_coords(info_1)
            info_2 = self.interpolate_repeated_coords(info_2)
            self.result, self.result_cords = align_info(
                info_1, info_2, no_speed_correction=no_alignment_speed_correction,
            )
            if self.result is None:
                print("Shifting lists a" "s list 2 starts from behind list 1")
                self.result, self.result_cords = align_info(
                    info_2,
                    info_1,
                    no_speed_correction=no_alignment_speed_correction,
                    check_reverse=False,
                )
                # Switch right and left again for correct output return
                self.result, self.result_cords = (
                    self.result[:, ::-1],
                    self.result_cords[:, ::-1],
                )
        elif format == "blackvue":
            assert (l_dataDir is not None) or (r_dataDir is not None), (
                "For this format, two valid directory paths " "are required "
            )
            self.leftImgList = [
                str(img) for img in sorted(pathlib.Path(l_dataDir).rglob("*g"))
            ]
            self.rightImgList = [
                str(img) for img in sorted(pathlib.Path(r_dataDir).rglob("*g"))
            ]
            self.result = self.aligned_lists(
                self.leftImgList,
                self.rightImgList,
                no_speed_correction=no_alignment_speed_correction,
            )
            if self.result is None:
                print("Shifting lists as list 2 starts from behind list 1")
                self.result = self.aligned_lists(
                    self.rightImgList,
                    self.leftImgList,
                    no_speed_correction=no_alignment_speed_correction,
                )[:, ::-1]
        self.my_transforms = []
        if augmentation:
            if height and width:
                self.my_transforms.append(transforms.Resize((height, width)))
            self.my_transforms.append(
                transforms.ColorJitter(
                    brightness=0.5, contrast=0.5, saturation=0, hue=0
                )
            )
            self.my_transforms.append(transforms.RandomAffine(0, (0.05, 0.05)))
        self.my_transforms.append(transforms.ToTensor())
        if augmentation:
            self.my_transforms.append(
                transforms.Normalize((0.485, 0.456, 0.456), (0.229, 0.224, 0.225))
            )
            self.my_transforms.append(transforms.RandomErasing(p=0))
        self.transform = transforms.Compose(self.my_transforms)

    @staticmethod
    def aligned_lists(img_list_1, img_list_2, no_speed_correction=False, max_dist=250):
        final_list = []
        for m in tqdm(range(len(img_list_1))):
            img_1_path = img_list_1[m]
            if not ((".png" in img_1_path) or (".jpg" in img_1_path)):
                continue
            prev_distance = None
            select = 0
            if len(img_list_2) <= 1:
                break
            for n, img_2_path in enumerate(img_list_2):
                if not ((".png" in img_2_path) or (".jpg" in img_2_path)):
                    continue
                l_coord_strings, right_coord_strings = (
                    img_1_path.split("_"),
                    img_2_path.split("_"),
                )
                left_cord = np.array(
                    [float(l_coord_strings[-2]), float(l_coord_strings[-1][:-4])]
                )
                right_cord = np.array(
                    [
                        float(right_coord_strings[-2]),
                        float(right_coord_strings[-1][:-4]),
                    ]
                )
                dist = np.linalg.norm(left_cord - right_cord)
                if prev_distance is None:
                    prev_distance = dist
                elif prev_distance <= dist:
                    select += 1
                else:
                    prev_distance = dist
                if select and m == 0 and n == 1:
                    # Reverse the lists. Reference list (`info_1`) should start behind
                    return None
                if select:
                    final_list.append((img_list_1[m], img_list_2[n - 1]))
                    dist_m = vincenty(left_cord, right_cord).meters
                    if no_speed_correction:
                        img_list_2 = img_list_2[n - 1 :]
                    elif (
                        dist_m > max_dist
                    ):  # if distance of the frames is more than max_dist [m] skip to next frame
                        continue
                    else:
                        pass
                    break
        return np.array(final_list)

    @staticmethod
    def interpolate_repeated_coords(info):
        # info: [n_images x [img_path, lat, lon, alt]
        n_images = info.shape[0]
        prev_x, prev_y = None, None
        indexes, unique_coords_x, unique_coords_y = [], [], []
        for n, info_row in enumerate(info):
            _, x, y, _ = info_row
            if x != prev_x or y != prev_y:
                indexes.append(n)
                unique_coords_x.append(float(x))
                unique_coords_y.append(float(y))
            prev_x, prev_y = x, y
        new_coords_x = np.interp(range(n_images), indexes, unique_coords_x)
        new_coords_y = np.interp(range(n_images), indexes, unique_coords_y)
        info[:, 1] = new_coords_x
        info[:, 2] = new_coords_y
        return info

    def __len__(self):
        return len(self.result)

    def __getitem__(self, index):
        image1 = Image.open(self.result[index][0]).convert("RGB")
        image2 = Image.open(self.result[index][1]).convert("RGB")

        image1 = self.transform(image1)
        image2 = self.transform(image2)
        return image1, image2


def read_text_file_aiim(txt_path):
    assert txt_path is not None, "No path provided for txt file"
    info = []
    with open(txt_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        for row in csv_reader:
            if "GUID" not in row:
                info.append([row[1]] + row[3:6])
    return np.array(info)


def read_text_file_fangzhou(txt_path):
    assert txt_path is not None, "No path provided for txt file"
    info = []
    dirname = os.path.dirname(txt_path)
    image_dir = os.path.abspath(os.path.join(dirname, "..", "image"))
    with open(txt_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=" ")
        start_read = False
        for row in csv_reader:
            if not start_read:
                if "SeqNum" in row:
                    start_read = True
            else:
                row_processed = []
                for entry in row:
                    if entry:
                        row_processed.append(entry)
                image_path = os.path.join(image_dir, row_processed[1] + ".jpg")
                info.append(
                    [image_path] + row_processed[2:4] + [0]
                )  # Also add altitude at the end
    return np.array(info)


def get_lat_lon_alt(info):
    lat_lon_alt = np.array(info[1:4]).astype(float)
    return lat_lon_alt


def align_info(
    info_1,
    info_2,
    no_speed_correction=True,
    distance_threshold=0.05,
    check_reverse=True,
):
    # info: [n_images x [img_path, lat, lon, alt]
    final_list, final_cords = [], []
    for m in tqdm(range(info_1.shape[0])):
        info_1_row = info_1[m]
        coords_left = get_lat_lon_alt(info_1_row)
        prev_distance = None
        prev_distance_smaller = False
        if len(info_2) <= 1:
            break
        prev_index = 0
        for n, info_2_row in enumerate(info_2):
            coords_right = get_lat_lon_alt(info_2_row)
            dist = np.linalg.norm(coords_left - coords_right)
            if dist > distance_threshold:
                info_2 = info_2[1:]
            else:
                if prev_distance is None:
                    prev_distance = dist
                elif prev_distance <= dist:
                    prev_distance_smaller = True
                else:
                    prev_distance = dist
                if check_reverse and prev_distance_smaller and m == 0 and n == 1:
                    # Reverse the lists. Reference list (`info_1`) should start behind
                    return None, None
                if prev_distance_smaller:
                    # Get index of previous image
                    if prev_index < info_2.shape[0]:
                        prev_index = n - 1
                    else:
                        prev_index = -1
                    coords_right = get_lat_lon_alt(info_2[prev_index, :])
                    final_list.append(
                        (info_1_row[0], info_2[prev_index, 0])
                    )  # Add filenames
                    final_cords.append((coords_left, coords_right))  # Add lat, lon, alt
                    if no_speed_correction:
                        info_2 = info_2[n:]
                    else:
                        pass
                    break
    return np.array(final_list), np.array(final_cords)


# test
if __name__ == "__main__":
    import cv2

    data_dir = "/data/projects/crd_project/test_dataset/comparison_essen/_data_aiim_change_detection_comparison00-essen_/"
    crd_loader = Loader(
        txt_info_2=os.path.join(data_dir, "20190701", "2019-07-011219_GPS_Photo.txt"),
        txt_info_1=os.path.join(data_dir, "20190826", "2019-08-261150_GPS_Photo.txt"),
        no_alignment_speed_correction=False,
    )
    trainLoader = DataLoader(crd_loader, batch_size=1, shuffle=False, num_workers=8)

    for img1, img2 in trainLoader:
        img_a = np.transpose(img1.numpy()[0], axes=[1, 2, 0])
        img_b = np.transpose(img2.numpy()[0], axes=[1, 2, 0])
        plt.imshow(np.hstack([img_a, img_b]))
        plt.show()
