import torch
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import os
import sys

sys.path.append("./text_reader")
from detector_model import EAST
import reader
from dataset import get_rotate_mat, AlignCollate
import numpy as np
import lanms
import cv2
import argparse
import string
import glob, time


def make_rect_upright(rects_points, relax=0.0):
    """
    Takes a set of bbox points (n, 4, 2) and returns upright bboxes
    :param points: (n, 4, 2) numpy array of (possibly) rotated bboxes
    :param relax: a float to define by what factor to relax the bbox height and width
    :return: (n, 4, 2) numpy array of upright bboxes, (n, 4) rectangles
    """
    final_points, final_rects = [], []
    for rect_points in rects_points:
        curr_rect = cv2.boundingRect(rect_points)
        pts = (
            (curr_rect[0] - relax * curr_rect[2], curr_rect[1] - relax * curr_rect[3]),
            (curr_rect[0] - relax * curr_rect[2], curr_rect[1] + curr_rect[3]),
            (curr_rect[0] + curr_rect[2], curr_rect[1] + curr_rect[3]),
            (curr_rect[0] + curr_rect[2], curr_rect[1] - relax * curr_rect[3]),
        )
        pts = np.array(pts).astype(np.int32)
        final_points.append(pts)
        final_rects.append(curr_rect)
    return np.array(final_points), np.array(final_rects)


def resize_img(img):
    """resize image to be divisible by 32"""
    w, h = img.size
    resize_w = w
    resize_h = h

    resize_h = resize_h if resize_h % 32 == 0 else int(resize_h / 32) * 32
    resize_w = resize_w if resize_w % 32 == 0 else int(resize_w / 32) * 32
    img = img.resize((resize_w, resize_h), Image.BILINEAR)
    ratio_h = resize_h / h
    ratio_w = resize_w / w

    return img, ratio_h, ratio_w


def load_pil(img):
    """convert PIL Image to torch.Tensor"""
    t = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )
    return t(img).unsqueeze(0)


def is_valid_poly(res, score_shape, scale):
    """check if the poly in image scope
    Input:
            res        : restored poly in original image
            score_shape: score map shape
            scale      : feature map -> image
    Output:
            True if valid
    """
    cnt = 0
    for i in range(res.shape[1]):
        if (
            res[0, i] < 0
            or res[0, i] >= score_shape[1] * scale
            or res[1, i] < 0
            or res[1, i] >= score_shape[0] * scale
        ):
            cnt += 1
    return True if cnt <= 1 else False


def restore_polys(valid_pos, valid_geo, score_shape, scale=4):
    """restore polys from feature maps in given positions
    Input:
            valid_pos  : potential text positions <numpy.ndarray, (n,2)>
            valid_geo  : geometry in valid_pos <numpy.ndarray, (5,n)>
            score_shape: shape of score map
            scale      : image / feature map
    Output:
            restored polys <numpy.ndarray, (n,8)>, index
    """
    polys = []
    index = []
    valid_pos *= scale
    d = valid_geo[:4, :]  # 4 x N
    angle = valid_geo[4, :]  # N,

    for i in range(valid_pos.shape[0]):
        x = valid_pos[i, 0]
        y = valid_pos[i, 1]
        y_min = y - d[0, i]
        y_max = y + d[1, i]
        x_min = x - d[2, i]
        x_max = x + d[3, i]
        rotate_mat = get_rotate_mat(-angle[i])

        temp_x = np.array([[x_min, x_max, x_max, x_min]]) - x
        temp_y = np.array([[y_min, y_min, y_max, y_max]]) - y
        coordidates = np.concatenate((temp_x, temp_y), axis=0)
        res = np.dot(rotate_mat, coordidates)
        res[0, :] += x
        res[1, :] += y

        if is_valid_poly(res, score_shape, scale):
            index.append(i)
            polys.append(
                [
                    res[0, 0],
                    res[1, 0],
                    res[0, 1],
                    res[1, 1],
                    res[0, 2],
                    res[1, 2],
                    res[0, 3],
                    res[1, 3],
                ]
            )
    return np.array(polys), index


def get_boxes(score, geo, score_thresh=0.9, nms_thresh=0.2):
    """get boxes from feature map
    Input:
            score       : score map from model <numpy.ndarray, (1,row,col)>
            geo         : geo map from model <numpy.ndarray, (5,row,col)>
            score_thresh: threshold to segment score map
            nms_thresh  : threshold in nms
    Output:
            boxes       : final polys <numpy.ndarray, (n,9)>
    """
    score = score[0, :, :]
    xy_text = np.argwhere(score > score_thresh)  # n x 2, format is [r, c]
    if xy_text.size == 0:
        return None

    xy_text = xy_text[np.argsort(xy_text[:, 0])]
    valid_pos = xy_text[:, ::-1].copy()  # n x 2, [x, y]
    valid_geo = geo[:, xy_text[:, 0], xy_text[:, 1]]  # 5 x n
    polys_restored, index = restore_polys(valid_pos, valid_geo, score.shape)
    if polys_restored.size == 0:
        return None

    boxes = np.zeros((polys_restored.shape[0], 9), dtype=np.float32)
    boxes[:, :8] = polys_restored
    boxes[:, 8] = score[xy_text[index, 0], xy_text[index, 1]]
    boxes = lanms.merge_quadrangle_n9(boxes.astype("float32"), nms_thresh)
    return boxes


def adjust_ratio(boxes, ratio_w, ratio_h):
    """refine boxes
    Input:
            boxes  : detected polys <numpy.ndarray, (n,9)>
            ratio_w: ratio of width
            ratio_h: ratio of height
    Output:
            refined boxes
    """
    if boxes is None or boxes.size == 0:
        return None
    boxes[:, [0, 2, 4, 6]] /= ratio_w
    boxes[:, [1, 3, 5, 7]] /= ratio_h
    return np.around(boxes)


def detect(img, model, device):
    """detect text regions of img using model
    Input:
            img   : PIL Image
            model : detection model
            device: gpu if gpu is available
    Output:
            detected polys
    """
    img, ratio_h, ratio_w = resize_img(img)
    with torch.no_grad():
        score, geo = model(load_pil(img).to(device))
    boxes = get_boxes(score.squeeze(0).cpu().numpy(), geo.squeeze(0).cpu().numpy())
    return adjust_ratio(boxes, ratio_w, ratio_h)


def plot_boxes(img, boxes):
    """plot boxes on image"""
    if boxes is None:
        return img

    draw = ImageDraw.Draw(img)
    for box in boxes:
        draw.polygon(
            [box[0], box[1], box[2], box[3], box[4], box[5], box[6], box[7]],
            outline=(0, 255, 0),
        )
    return img


def detect_dataset(model, device, test_img_path, submit_path):
    """detection on whole dataset, save .txt results in submit_path
    Input:
            model        : detection model
            device       : gpu if gpu is available
            test_img_path: dataset path
            submit_path  : submit result for evaluation
    """
    img_files = os.listdir(test_img_path)
    img_files = sorted(
        [os.path.join(test_img_path, img_file) for img_file in img_files]
    )

    for i, img_file in enumerate(img_files):
        print("evaluating {} image".format(i), end="\r")
        boxes = detect(Image.open(img_file), model, device)
        seq = []
        if boxes is not None:
            seq.extend(
                [",".join([str(int(b)) for b in box[:-1]]) + "\n" for box in boxes]
            )
        with open(
            os.path.join(
                submit_path, "res_" + os.path.basename(img_file).replace(".jpg", ".txt")
            ),
            "w",
        ) as f:
            f.writelines(seq)


class Config:
    def __init__(self):
        self.workers = 4
        # self.east_model_path = '/data/output/ahmed-badar/model_epoch_456.pth'
        self.east_model_path = "/data/input/models/ocr_models/pths/east_vgg16.pth"
        self.saved_model = (
            "/data/input/models/ocr_models/TPS-ResNet-BiLSTM-Attn-case-sensitive.pth"
        )
        self.dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.imgH = 32
        self.imgW = 100
        self.rgb = False
        self.character = string.printable[:-6]
        self.sensitive = True
        self.Transformation = "TPS"
        self.FeatureExtraction = "ResNet"
        self.SequenceModeling = "BiLSTM"
        self.Prediction = "Attn"
        self.num_fiducial = 20
        self.input_channel = 1
        self.output_channel = 512
        self.hidden_size = 256
        self.batch_max_length = 25
        self.model_east = EAST().to(self.dev)
        self.model_east.load_state_dict(torch.load(self.east_model_path))
        self.recog_model = reader.get_model(self)


def read_crop(img, opt):
    # model_path = opt.east_model_path
    # # res_img     = './image_1.png'
    # model_east = EAST().to(dev)
    # if opt.sensitive:
    # 	opt.character = string.printable[:-6]
    # model_east.load_state_dict(torch.load(model_path))
    # recog_model = reader.get_model(opt)
    img = img.resize((256, 128), 3)
    # img.show()
    # input('wait')
    model_east = opt.model_east
    recog_model = opt.recog_model
    model_east.eval()
    recog_model.eval()
    align = AlignCollate(imgH=opt.imgH, imgW=opt.imgW)
    crops, centers = [], []
    boxes = detect(img, model_east, opt.dev)
    if boxes is not None:
        for box in boxes:
            crop = [int(box[i]) for i in (0, 1, 4, 5)]
            crop[0], crop[1], crop[2], crop[3] = crop[0], crop[1], crop[2], crop[3]
            # ImageDraw.Draw(img).rectangle(crop, fill=None, outline=(255, 255, 0))
            new_img = img.crop(crop)
            crops.append(new_img.convert("L"))
            centers.append(((crop[0] + crop[2]) // 2, (crop[1] + crop[3]) // 2))
        # img.show()
        new_imgs = align(crops)
        # lines = reader.read(opt, new_imgs, recog_model)
        lines, confs, cents = reader.read_with_thresh(
            opt, new_imgs, recog_model, centers
        )
        return lines, confs, cents
    else:
        return []
    # for n, line in enumerate(lines):
    # 	curr_line = line.split("[s]", 1)[0]
    # 	ImageDraw.Draw(img).text(boxes[n, :2] - 10, curr_line, font=font, fill="red")
    # img.show()


if __name__ == "__main__":

    config = Config()
    image = Image.open("/volumes2/text_datasets/MLT_NIE/train_img/tr_img_04360.jpg")
    read_crop(image, config)
