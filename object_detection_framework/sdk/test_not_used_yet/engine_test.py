from shutil import rmtree
import os
import sys
import unittest

from contextlib import redirect_stdout
import io
import json
import numpy as np
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../.."))
# pylint: disable=wrong-import-position
import sdk.src.constants as const
from sdk.src.engine import Engine
import sdk.src.utils as utils

# pylint: enable=wrong-import-position

# pylint: disable=protected-access

CKPT_FILE = (
    "/data/nie/teams/arl/projects/zoo/detection/ssd/voc/resnet18/20.03/model_final.pth"
)
CONFIG_FILE = "/data/nie/teams/arl/projects/zoo/detection/ssd/voc/resnet18/20.03/resnet18_ssd512_voc0712.yaml"
EXPECTED_IMG = "/data/nie/teams/arl/system_tests_data/object_detection_framework/sdk/ref/000009.jpg"
EXPECTED_JSON = "/data/nie/teams/arl/system_tests_data/object_detection_framework/sdk/ref/000009.json"
EXPECTED_JSON_NIE = "/data/nie/teams/arl/system_tests_data/object_detection_framework/sdk/ref/000009_nie.json"
EXPECTED_TXT = "/data/nie/teams/arl/system_tests_data/object_detection_framework/sdk/ref/000009.txt"
EXPECTED_XML = "/data/nie/teams/arl/system_tests_data/object_detection_framework/sdk/ref/000009.xml"
INPUT_IMG = "/data/nie/teams/arl/system_tests_data/object_detection_framework/test_data/voc/000009.jpg"
INPUT_IMG_UNSUPPORTED = (
    "/data/nie/teams/arl/system_tests_data/object_detection_framework/test_data"
    "/mix_jpg_JPG_png_PNG/img9.PNG"
)
INPUT_DIR = "/data/nie/teams/arl/system_tests_data/object_detection_framework/test_data/mix_jpg_JPG_png_PNG"
INPUT_DIR_NO_JPG = (
    "/data/nie/teams/arl/system_tests_data/object_detection_framework/test_data"
    "/mix_jpg_JPG_png_PNG/no_jpg"
)


class EngineTestCase(unittest.TestCase):
    def test_init_default__members_set(self):
        eng = Engine()

        self.assertEqual(eng.config_file, const.DEFAULT_CFG)
        self.assertEqual(eng.config_options, const.DEFAULT_CFG_OPTIONS)
        self.assertEqual(eng.ckpt, const.DEFAULT_CKPT)
        self.assertEqual(eng.dataset_type, const.DEFAULT_DATASET_TYPE)
        self.assertEqual(eng.score_threshold, const.DEFAULT_SCORE_THRESHOLD)
        self.assertEqual(eng.output_dir, const.DEFAULT_OUTPUT_DIR)
        self.assertEqual(eng.output_format, const.DEFAULT_OUTPUT_FORMAT)
        self.assertEqual(eng.output_write, const.DEFAULT_OUTPUT_WRITE)


class LoadModelTestCase(unittest.TestCase):
    def test_load_model__data_members_set(self):
        eng = Engine(config_file=CONFIG_FILE, ckpt=CKPT_FILE)
        eng.load_model()

        self.assertIsNotNone(eng._class_names)
        self.assertIsNotNone(eng._device)
        self.assertIsNotNone(eng._model)
        self.assertIsNotNone(eng._transforms)


class InferImgTestCase(unittest.TestCase):
    def setUp(self):
        if os.path.exists(const.DEFAULT_OUTPUT_DIR):
            rmtree(const.DEFAULT_OUTPUT_DIR)
        self.eng = Engine(config_file=CONFIG_FILE, ckpt=CKPT_FILE)
        self.eng.load_model()

    def test_infer_img_fmt_img__is_numpy_and_has_correct_shape(self):
        self.eng.output_format = "img"
        result = self.eng.infer_img(INPUT_IMG)
        expected = np.array(Image.open(EXPECTED_IMG))
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, expected.shape)

    def test_infer_img_fmt_json__correct_format(self):
        self.eng.output_format = "json"
        result_str = self.eng.infer_img(INPUT_IMG)
        result = json.loads(result_str)

        with open(EXPECTED_JSON, mode="r") as file:
            expected = json.load(file)

        self.assertDictEqual(result, expected)

    def test_infer_img_fmt_json_nie__correct_format(self):
        self.eng.output_format = "json_nie"
        result_str = self.eng.infer_img(INPUT_IMG)
        result = json.loads(result_str)

        with open(EXPECTED_JSON_NIE, mode="r") as file:
            expected = json.load(file)

        self.assertDictEqual(result, expected)

    def test_infer_img_fmt_txt__correct_format(self):
        self.eng.output_format = "txt"
        result = self.eng.infer_img(INPUT_IMG)

        with open(EXPECTED_TXT, mode="r") as file:
            expected = file.read()
            self.assertMultiLineEqual(result, expected)

    def test_infer_img_fmt_xml__correct_format(self):
        self.eng.output_format = "xml"
        result = self.eng.infer_img(INPUT_IMG)

        with open(EXPECTED_XML, mode="r") as file:
            expected = file.read()
            self.assertMultiLineEqual(result, expected)

    def test_infer_img_non_existent__print_message_and_nothing_predicted(self):
        filename = "non_existing_image.jpg"
        io_string = io.StringIO()
        with redirect_stdout(io_string):
            result = self.eng.infer_img(filename)
        out = io_string.getvalue()
        expected = f"Warning: {filename} does not exist!"
        self.assertIn(expected, out)
        self.assertIsNone(result)
        output_images_path = utils.find_files_with_extensions(
            const.DEFAULT_OUTPUT_DIR, const.SUPPORTED_IMAGE_FORMATS
        )
        output_images = [os.path.basename(image) for image in output_images_path]
        self.assertEqual(0, len(output_images))

    def test_infer_img_unsupported_format__print_message_and_nothing_predicted(self):
        io_string = io.StringIO()
        with redirect_stdout(io_string):
            result = self.eng.infer_img(INPUT_IMG_UNSUPPORTED)
        out = io_string.getvalue()
        expected = f"{INPUT_IMG_UNSUPPORTED} is not supported. Supported formats: {const.SUPPORTED_IMAGE_FORMATS}!"
        self.assertIn(expected, out)
        self.assertIsNone(result)
        output_images_path = utils.find_files_with_extensions(
            const.DEFAULT_OUTPUT_DIR, const.SUPPORTED_IMAGE_FORMATS
        )
        output_images = [os.path.basename(image) for image in output_images_path]
        self.assertEqual(0, len(output_images))


class InferTestCase(unittest.TestCase):
    def setUp(self):
        if os.path.exists(const.DEFAULT_OUTPUT_DIR):
            rmtree(const.DEFAULT_OUTPUT_DIR)
        self.eng = Engine(config_file=CONFIG_FILE, ckpt=CKPT_FILE)

    def test_infer_dir_without_write__raise_exception(self):
        self.eng.output_write = False
        self.eng.load_model()

        with self.assertRaises(ValueError) as context:
            self.eng.infer(INPUT_DIR)
        self.assertEqual(
            "output_write is False. Can't run inference on directory without saving outputs!",
            context.exception.args[0],
        )

    def test_infer_dir_with_mix_extensions__only_jpg_case_insensitive_predicted(self):
        self.eng.load_model()
        self.eng.infer(INPUT_DIR)
        input_images_path = utils.find_files_with_extensions(
            INPUT_DIR, const.SUPPORTED_IMAGE_FORMATS
        )
        output_images_path = utils.find_files_with_extensions(
            const.DEFAULT_OUTPUT_DIR, const.SUPPORTED_IMAGE_FORMATS
        )
        input_images = [os.path.basename(image) for image in input_images_path]
        output_images = [os.path.basename(image) for image in output_images_path]
        self.assertCountEqual(input_images, output_images)

    def test_infer_no_jpg_found__print_message_and_nothing_predicted(self):
        self.eng.load_model()
        io_string = io.StringIO()
        with redirect_stdout(io_string):
            self.eng.infer(INPUT_DIR_NO_JPG)
        out = io_string.getvalue()
        expected = (
            f"Warning: no image found with extension {const.SUPPORTED_IMAGE_FORMATS}!"
        )
        self.assertIn(expected, out)
        output_images_path = utils.find_files_with_extensions(
            const.DEFAULT_OUTPUT_DIR, const.SUPPORTED_IMAGE_FORMATS
        )
        self.assertEqual(len(output_images_path), 0)


if __name__ == "__main__":
    unittest.main()
