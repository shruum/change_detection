import torch
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
from detector_model import EAST
import reader
import os
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", required=True, help="Image directory")
    parser.add_argument(
        "--workers", type=int, help="number of data loading workers", default=4
    )
    parser.add_argument("--batch_size", type=int, default=192, help="input batch size")
    parser.add_argument(
        "--saved_model",
        default="/data/output/ahmed-badar/model_epoch_557.pth",
        required=True,
        help="path to saved_model for recognition",
    )
    parser.add_argument(
        "--east_model_path",
        default="/data/output/ahmed-badar/TPS-ResNet-BiLSTM-Attn-case-sensitive.pth",
        help="path to saved_model for east",
    )
    """ Data processing """
    parser.add_argument(
        "--batch_max_length", type=int, default=25, help="maximum-label-length"
    )
    parser.add_argument(
        "--imgH", type=int, default=32, help="the height of the input image"
    )
    parser.add_argument(
        "--imgW", type=int, default=100, help="the width of the input image"
    )
    parser.add_argument("--rgb", action="store_true", help="use rgb input")
    parser.add_argument(
        "--character",
        type=str,
        default="0123456789abcdefghijklmnopqrstuvwxyz",
        help="character label",
    )
    parser.add_argument(
        "--sensitive", action="store_true", help="for sensitive character mode"
    )
    parser.add_argument(
        "--PAD",
        action="store_true",
        help="whether to keep ratio then pad for image resize",
    )
    """ Model Architecture """
    parser.add_argument(
        "--Transformation",
        type=str,
        required=True,
        help="Transformation stage. None|TPS",
    )
    parser.add_argument(
        "--FeatureExtraction",
        type=str,
        required=True,
        help="FeatureExtraction stage. VGG|RCNN|ResNet",
    )
    parser.add_argument(
        "--SequenceModeling",
        type=str,
        required=True,
        help="SequenceModeling stage. None|BiLSTM",
    )
    parser.add_argument(
        "--Prediction", type=str, required=True, help="Prediction stage. CTC|Attn"
    )
    parser.add_argument(
        "--num_fiducial",
        type=int,
        default=20,
        help="number of fiducial points of TPS-STN",
    )
    parser.add_argument(
        "--input_channel",
        type=int,
        default=1,
        help="the number of input channel of Feature extractor",
    )
    parser.add_argument(
        "--output_channel",
        type=int,
        default=512,
        help="the number of output channel of Feature extractor",
    )
    parser.add_argument(
        "--hidden_size", type=int, default=256, help="The size of the LSTM hidden state"
    )
    parser.add_argument(
        "--save_dir", type=str, default="result/", help="Path to save result images."
    )

    opt = parser.parse_args()
    # img_path    = '../ICDAR_2015/test_img/img_2.jpg'
    # img_path    = '/data/users/ahmed.badar/Downloads/image_1.png'
    model_path = "/data/output/ahmed-badar/model_epoch_456.pth"
    # model_path = opt.east_model_path
    # res_img     = './image_1.png'
    save_dir = opt.save_dir
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_east = EAST().to(device)
    if opt.sensitive:
        opt.character = string.printable[:-6]
    model_east.load_state_dict(torch.load(model_path))
    recog_model = reader.get_model(opt)
    model_east.eval()
    recog_model.eval()
    img_paths = sorted(glob.glob(opt.image_dir + "*g"))
    start_time = time.time()
    align = AlignCollate(imgH=32, imgW=100)
    font = ImageFont.truetype(
        "/volumes2/fonts/ZCOOL_XiaoWei/ZCOOLXiaoWei-Regular.ttf", 20
    )
    for img_path in img_paths:
        img = Image.open(img_path)
        tol = 0.1
        boxes = detect(img, model_east, device)
        if boxes is None:
            continue
        # cv_boxes = boxes[:, :8]
        crops = []
        for box in boxes:
            crop = [int(box[i]) for i in (0, 1, 4, 5)]
            crop[0], crop[1], crop[2], crop[3] = (
                crop[0] - 3,
                crop[1] - 3,
                crop[2] + 5,
                crop[3] + 5,
            )
            ImageDraw.Draw(img).rectangle(crop, fill=None, outline=(255, 255, 0))
            new_img = img.crop(crop)
            crops.append(new_img.convert("L"))
        new_imgs = align(crops)
        lines = reader.read(opt, new_imgs, recog_model)
        for n, line in enumerate(lines):
            curr_line = line.split("[s]", 1)[0]
            ImageDraw.Draw(img).text(
                boxes[n, :2] - 10, curr_line, font=font, fill="red"
            )
        img.save(save_dir + os.path.basename(img_path))
