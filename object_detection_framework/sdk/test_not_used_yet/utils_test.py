import os
import sys
import unittest


sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../.."))
# pylint: disable=wrong-import-position
import sdk.src.utils as utils

# pylint: enable=wrong-import-position

BACKGROUND_CLASS_NAME = "__background__"
COCO_NUMBER_OF_CLASSES = 81
VOC_NUMBER_OF_CLASSES = 21


class GetClassNamesTestCase(unittest.TestCase):
    def test_coco_dataset__correct_number_of_names(self):
        class_names = utils.get_class_names("coco")
        self.assertEqual(len(class_names), COCO_NUMBER_OF_CLASSES)
        self.assertEqual(class_names[-1], "toothbrush")

    def test_voc_dataset__correct_number_of_names(self):
        class_names = utils.get_class_names("voc")
        self.assertEqual(len(class_names), VOC_NUMBER_OF_CLASSES)
        self.assertEqual(class_names[-1], "tvmonitor")

    def test_invalid_dataset__raise_exception(self):
        with self.assertRaises(NotImplementedError) as context:
            utils.get_class_names("invalid_dataset")
        self.assertEqual(
            "invalid_dataset dataset not supported.", context.exception.args[0]
        )

    def test_include_background__background_in_class_names(self):
        class_names = utils.get_class_names("coco", include_background=True)
        self.assertIn(BACKGROUND_CLASS_NAME, class_names)

    def test_not_include_background__background_not_in_class_names(self):
        class_names = utils.get_class_names("coco", include_background=False)
        self.assertNotIn(BACKGROUND_CLASS_NAME, class_names)


if __name__ == "__main__":
    unittest.main()
