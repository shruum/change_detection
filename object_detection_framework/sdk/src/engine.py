import os
import sys
import time

import numpy as np
from PIL import Image
import torch

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../.."))
# pylint: disable=wrong-import-position
from od.data.transforms import build_transforms
from od.modeling.detector import build_detection_model
from od.utils import mkdir
from od.utils.checkpoint import CheckPointer
from od.utils.detections_processor import convert_to_format

import sdk.src.constants as const
import sdk.src.utils as utils

# pylint: enable=wrong-import-position


class Engine:
    def __init__(
        self,
        config_file=const.DEFAULT_CFG,
        config_options=const.DEFAULT_CFG_OPTIONS,
        ckpt=const.DEFAULT_CKPT,
        dataset_type=const.DEFAULT_DATASET_TYPE,
        score_threshold=const.DEFAULT_SCORE_THRESHOLD,
        output_format=const.DEFAULT_OUTPUT_FORMAT,
        output_write=const.DEFAULT_OUTPUT_WRITE,
        verbose=const.DEFAULT_VERBOSE,
    ):
        """
        Provides the functionality to run inference on a single image or a directory with images in JPG format.
        :param config_file: Path to the configuration file.
        :param config_options: Configuration options that overwrites those from the configuration file.
        :param ckpt: Path to the checkpoint/model file.
        :param dataset_type: 'coco' / 'voc'.
        :param score_threshold: Threshold to filter the detections.
        :param output_format: 'img' / 'json' / 'json_nie' / 'txt' / 'xml'.
        :param output_write: If True the result is save in the output_dir, otherwise the result object will be returned.
        :param verbose: Print logs.
        """
        self.config_file = config_file
        self.config_options = config_options
        self.ckpt = ckpt
        self.dataset_type = dataset_type
        self.score_threshold = score_threshold
        self.output_format = output_format
        self.output_write = output_write
        self.verbose = verbose

        self._class_names = None
        self._device = None
        self._model = None
        self._transforms = None

    @torch.no_grad()
    def load_model(self):
        """Prepare the model for inference."""
        config = utils.config_load(self.config_file, self.config_options)
        if self.verbose:
            print("*" * 80)
            print(f"Loaded configuration file {self.config_file}")
            with open(self.config_file, "r") as config_file:
                print(config_file.read())
            print("*" * 80)
            print(f"Running with config:\n{config}")
            print("*" * 80)

        self._class_names = utils.get_class_names(self.dataset_type)
        self.include_background = config.DATA_LOADER.INCLUDE_BACKGROUND

        self._device = torch.device(config.MODEL.DEVICE)
        self._model = build_detection_model(config)
        self._model = self._model.to(self._device)
        check_pointer = CheckPointer(self._model, save_dir=config.OUTPUT_DIR)
        check_pointer.load(self.ckpt, use_latest=self.ckpt is None)
        weight_file = self.ckpt if self.ckpt else check_pointer.get_checkpoint_file()
        self._transforms = build_transforms(config, is_train=False)
        self._model.eval()
        if self.verbose:
            print(f"\nModel loaded with {weight_file} checkpoints.")

    def infer(self, path=const.DEFAULT_INPUT_DIR, output_dir=const.DEFAULT_OUTPUT_DIR):
        """
        Infer a single image or a directory with images based on the input type.
        :param path: The full name of an image or a path of a directory  with images.
        :return: - The detection result object in case the `input` is a single image.
                 - `None` in case no image with supported extension is found or the `path` is a directory.

        Note: The result(s) are saved in the `self.output_dir` also, in case `self.output_write` is True.
        Note: The name of the result is generated as: os.path.splitext(image_basename)[0] + '.' + self.output_format.
        Note: If the `input` parameter is a directory and the `self.output_write` is False `ValueError` is raised.
        """
        if os.path.isdir(path):
            mkdir(output_dir)
            if self.output_write is False:
                raise ValueError(
                    "output_write is False. Can't run inference on directory without saving outputs!"
                )
            image_paths = utils.find_files_with_extensions(
                path, const.SUPPORTED_IMAGE_FORMATS
            )
            image_count = len(image_paths)
            if not image_count:
                print(
                    f"\nWarning: no image found with extension {const.SUPPORTED_IMAGE_FORMATS}!"
                )
                return None

            if self.verbose:
                print(f"\nRun inference on {image_count} images from {path}:")

            for i, image_path in enumerate(image_paths):
                if self.verbose:
                    print(f"({(i + 1):04d}/{image_count:04d}) ", end="")
                self.infer_img(image_path, output_dir)
            return None

        if self.verbose:
            print(f"\nRun inference on {path} image:")
        return self.infer_img(path, output_dir)

    @torch.no_grad()
    def infer_img(self, image_path, output_dir):
        """
        Infer one image.
        :param image_path:
        :return: - The detection result object.
                 - `None` in case `image_path does not exist or has an extension that is not supported.

        Note: The result(s) are saved in the `self.output_dir` also, in case `self.output_write` is True.
        Note: The name of the result is generated as: os.path.splitext(image_basename)[0] + '.' + self.output_format.
        """
        start = time.time()
        if not os.path.exists(image_path):
            print(f"Warning: {image_path} does not exist!")
            return None

        if os.path.splitext(image_path)[1] not in const.SUPPORTED_IMAGE_FORMATS:
            print(
                f"{image_path} is not supported. Supported formats: {const.SUPPORTED_IMAGE_FORMATS}!"
            )
            return None

        image_basename = os.path.basename(image_path)

        image = np.array(Image.open(image_path).convert("RGB"))
        height, width = image.shape[:2]
        image_preprocessed = self._transforms(image)[0].unsqueeze(0)
        load_time = time.time() - start

        start = time.time()
        detection = self._model(image_preprocessed.to(self._device))[0]
        inference_time = time.time() - start

        detection = detection.resize((width, height)).to(torch.device("cpu")).numpy()
        boxes, labels, scores = (
            detection["boxes"],
            detection["labels"],
            detection["scores"],
        )

        indices = scores > self.score_threshold
        boxes = boxes[indices]
        labels = labels[indices].astype(int)
        if not self.include_background:
            labels += 1
        scores = scores[indices]
        meters = " | ".join(
            [
                f"objects {len(boxes):02d}",
                f"load {round(load_time * 1000):03d}ms",
                f"inference {round(inference_time * 1000):03d}ms",
                f"FPS {round(1.0 / inference_time)}",
            ]
        )

        if self.verbose:
            print(f"{image_basename}: {meters}")

        result = convert_to_format(
            self.output_format,
            image_path,
            image,
            boxes,
            labels,
            scores,
            self._class_names,
        )
        self.save_result(result, image_basename, output_dir)

        return result

    def save_result(self, result, image_basename, output_dir):
        if self.output_write:
            if self.output_format == "json_nie":
                output_basename = os.path.splitext(image_basename)[0] + "." + "json"
            else:
                output_basename = (
                    os.path.splitext(image_basename)[0] + "." + self.output_format
                )
            output_fullname = os.path.join(output_dir, output_basename)
            if self.output_format == "img":
                Image.fromarray(result).save(os.path.join(output_dir, image_basename))
            else:
                with open(output_fullname, mode="w") as file:
                    file.write(result)
