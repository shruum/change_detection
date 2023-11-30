import base64
import io
import json
import numpy as np
import os
import time
import torch
import torchvision.transforms as transform
from torch.nn import BatchNorm2d
from PIL import Image, ImageOps

from encoding_custom.models import get_segmentation_model
from encoding_custom.datasets.mapillary_commercial import MapillaryMerged

from .utils import get_mean_std, find_files_with_extensions

DEFAULT_DEVICE = 0
DEFAULT_MODEL = "rgpnet"
DEFAULT_BACKBONE = "resnet101"
DEFAULT_CKPT = "model/model_best.pth.tar"
DEFAULT_DATASET = "mapillary_merged"
DEFAULT_INPUT_DIR = "input"
DEFAULT_OUTPUT_DIR = "output"
DEFAULT_OUTPUT_FORMAT = "json"
DEFAULT_OUTPUT_WRITE = True
DEFAULT_VERBOSE = True
DEFAULT_MODEL_KWARGS = dict(
    aux=False,
    se_loss=False,
    pretrained=False,
    norm_layer=BatchNorm2d,
    base_size=1536,
    crop_size=1024,
)
SUPPORTED_IMAGE_FORMATS = [".jpg", ".JPG"]


class Engine:
    def __init__(
        self,
        model=DEFAULT_MODEL,
        backbone=DEFAULT_BACKBONE,
        ckpt=DEFAULT_CKPT,
        dataset=DEFAULT_DATASET,
        model_kwargs=DEFAULT_MODEL_KWARGS,
        output_dir=DEFAULT_OUTPUT_DIR,
        output_format=DEFAULT_OUTPUT_FORMAT,
        output_write=DEFAULT_OUTPUT_WRITE,
        verbose=DEFAULT_VERBOSE,
    ):
        """
        Provides the functionality to run inference on a single image or a directory with images in JPG format.
        :param ckpt: Path to the checkpoint/model file.
        :param dataset: 'mapillary_merged'
        :param output_dir: Output directory.
        :param output_format: 'json' / 'img'.
        :param output_write: If True the result is save in the output_dir, otherwise the result object will be returned.
        :param verbose: Print logs.
        """
        self.model_name = model
        self.backbone = backbone
        self.ckpt = ckpt
        self.dataset = dataset
        self.model_kwargs = model_kwargs
        self.output_dir = output_dir
        assert output_format in ["json", "img"], "unsupported output_format"
        self.output_format = output_format
        self.output_write = output_write
        self.verbose = verbose
        self.base_size = (
            None
            if "base_size" not in model_kwargs.keys()
            else model_kwargs["base_size"]
        )

        self._device = None
        self._model = None
        self._input_transform = None

    def set_gpu(self, gpu):
        self._device = torch.device("cpu" if gpu == -1 else "cuda:{}".format(gpu))

    @torch.no_grad()
    def load_model(self):
        """Prepare the model for inference."""
        self._model = get_segmentation_model(
            self.model_name,
            dataset=self.dataset,
            backbone=self.backbone,
            **self.model_kwargs,
        )

        checkpoint = torch.load(self.ckpt)
        if "state_dict" in checkpoint.keys():
            self._model.load_state_dict(checkpoint["state_dict"], strict=False)
        elif "model" in checkpoint.keys():
            self._model.load_state_dict(checkpoint["model"], strict=False)
        else:
            raise ("loaded checkpoint has no params key!")
        print(
            "=> loaded checkpoint '{}' (epoch {})".format(
                self.ckpt, checkpoint["epoch"]
            )
        )

        if self._device is None:
            self.set_gpu(DEFAULT_DEVICE)

        self._model.eval()
        self._model = self._model.to(self._device)

        if not os.path.exists(self.output_dir) and self.output_write:
            os.makedirs(self.output_dir)

        in_mean, in_std = get_mean_std(self.model_name, self.dataset)
        self._input_transform = transform.Compose(
            [transform.ToTensor(), transform.Normalize(in_mean, in_std)]
        )

    def infer(self, input=DEFAULT_INPUT_DIR):
        """
        Infer a single image or a directory with images based on the input type.
        :param input: The full name of an image or a path of a directory  with images.
        :return: The segmentation result object in case the `input` is a single image.

        Note: The result(s) are saved in the `self.output_dir` also, in case `self.output_write` is True.
        Note: The name of the result is generated as: os.path.splitext(image_basename)[0] + '.' + self.output_format .
        Note: If the `input` parameter is a directory and the `self.output_write` is False `ValueError` is raised.
        """
        if os.path.isdir(input):
            if self.output_write is False:
                raise ValueError(
                    "output_write is False. Can't run inference on directory without saving outputs!"
                )

            image_paths = find_files_with_extensions(input, SUPPORTED_IMAGE_FORMATS)
            image_count = len(image_paths)
            if not image_count:
                print(
                    f"\nWarning: no image found with extension {SUPPORTED_IMAGE_FORMATS}!"
                )
                return None

            if self.verbose:
                print(f"\nRun inference on {image_count} images from {input}:")

            for i, image_path in enumerate(image_paths):
                if self.verbose:
                    print(f"({(i + 1):04d}/{image_count:04d}) ", end="")
                self.infer_img(image_path)
        else:
            if self.verbose:
                print("Run inference on {} image:".format(input))
            return self.infer_img(input)

    @torch.no_grad()
    def infer_img(self, image_path):
        """Infer one image."""
        start = time.time()
        img = Image.open(image_path).convert("RGB")
        w_orig, h_orig = img.size
        img, padding = self._preprocess_image(img)
        img = self._input_transform(img)
        load_time = time.time() - start

        outputs = self._model(img.to(self._device).unsqueeze(0))[0]

        out_to_save_gpu = torch.max(outputs, 1)[1]
        out_to_save = out_to_save_gpu.cpu().numpy().squeeze()
        try:
            present_class_ids = out_to_save_gpu.unique().cpu().tolist()
        except RuntimeError:
            present_class_ids = set(out_to_save.flatten().tolist())

        out_to_save = Image.fromarray(np.asarray(out_to_save, np.uint8))
        out_to_save = ImageOps.expand(out_to_save, padding)
        out_to_save = out_to_save.resize((w_orig, h_orig), Image.NEAREST)
        infer_time = (time.time() - start) - load_time

        if self.output_write:
            out_path = os.path.join(
                self.output_dir, os.path.splitext(os.path.basename(image_path))[0]
            )
            if self.output_format == "img":
                out_to_save.save(out_path + ".png")
            elif self.output_format == "json":
                self._pil2json(out_to_save, out_path + ".json", present_class_ids)
            if self.verbose:
                save_time = time.time() - infer_time - load_time - start
                print(
                    f"[loadImage, infer, save] time: [{load_time:.3f}s, {infer_time:.3f}s, {save_time:.3f}s]"
                )
        else:
            if self.verbose:
                print(f"[loadImage, infer] time: [{load_time:.3f}s, {infer_time:.3f}s]")
            return out_to_save

    def _preprocess_image(self, image):
        if self.base_size is not None:
            image = self._resize(image, self.base_size)
        image, padding = self._crop2multiple_of_32(image)
        return image, padding

    @staticmethod
    def _resize(image, base_size):
        w_orig, h_orig = image.size
        if w_orig > h_orig:
            oh = base_size
            ow = int(1.0 * w_orig * oh / h_orig)
        else:
            ow = base_size
            oh = int(1.0 * h_orig * ow / w_orig)
        return image.resize((ow, oh), Image.BILINEAR)

    @staticmethod
    def _crop2multiple_of_32(image):
        w_n, h_n = image.size
        w_n = w_n // 32 * 32
        h_n = h_n // 32 * 32
        # center crop
        w, h = image.size
        x1 = int(round((w - w_n) / 2.0))
        y1 = int(round((h - h_n) / 2.0))
        padding = (x1, y1, w - w_n - x1, h - h_n - y1)
        return image.crop((x1, y1, x1 + w_n, y1 + h_n)), padding

    def _pil2json(self, out_to_save, out_path, present_class_ids):
        objects = self._create_objects_list(out_to_save, present_class_ids)
        raw_data = dict()
        raw_data["cv_task"] = 2
        raw_data["obj_num"] = len(objects)
        raw_data["output_type"] = 2
        raw_data["objects"] = objects
        raw_data["image"] = self._convert2base64(out_to_save)
        raw_data["ins"] = 0

        json_data = json.dumps(raw_data)
        with open(out_path, "w") as of:
            of.write(json_data)

    @staticmethod
    def _create_objects_list(img: Image, present_class_ids: list):
        class_names_dict = MapillaryMerged.CLASS_NAMES
        objects = [
            {"f_name": class_names_dict[cls_id], "f_code": cls_id, "color": [cls_id],}
            for cls_id in present_class_ids
        ]
        return objects

    @staticmethod
    def _convert2base64(out_to_save):
        byte_content = io.BytesIO()
        out_to_save.save(byte_content, format="PNG")
        base64_bytes = base64.b64encode(byte_content.getvalue())
        base64_string = base64_bytes.decode("utf-8")
        return base64_string
