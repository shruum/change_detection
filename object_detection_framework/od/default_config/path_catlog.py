import os
from od.data.datasets.dataset_class_names import dataset_classes


class DatasetCatalog:
    DATA_DIR = "/input/datasets"
    DATASETS = {
        "voc_2007_train": {"data_dir": "VOC2007", "split": "train"},
        "voc_2007_val": {"data_dir": "VOC2007", "split": "val"},
        "voc_2007_trainval": {"data_dir": "VOC2007", "split": "trainval"},
        "voc_2007_test": {"data_dir": "VOC2007", "split": "test"},
        "voc_2007_test_temp": {"data_dir": "VOC2007", "split": "test_temp"},
        "voc_2012_train": {"data_dir": "VOC2012", "split": "train"},
        "voc_2012_val": {"data_dir": "VOC2012", "split": "val"},
        "voc_2012_trainval": {"data_dir": "VOC2012", "split": "trainval"},
        "voc_2012_test": {"data_dir": "VOC2012", "split": "test"},
        "coco_2014_valminusminival": {
            "data_dir": "val2014",
            "ann_file": "annotations/instances_valminusminival2014.json",
        },
        "coco_2014_minival": {
            "data_dir": "val2014",
            "ann_file": "annotations/instances_minival2014.json",
        },
        "coco_2014_train": {
            "data_dir": "train2014",
            "ann_file": "annotations/instances_train2014.json",
        },
        "coco_2014_val": {
            "data_dir": "val2014",
            "ann_file": "annotations/instances_val2014.json",
        },
        "bdd_2018_train": {
            "data_dir": "images/train2018",
            "ann_file": "annotations/instances_train2018.json",
        },
        "bdd_2018_val": {
            "data_dir": "images/val2018",
            "ann_file": "annotations/instances_val2018.json",
        },
        "had_2018_train": {
            "data_dir": "train",
            "ann_file": "annotations/had/instances_train2018.json",
        },
        "had_2018_val": {
            "data_dir": "test/positives",
            "ann_file": "annotations/had/instances_val2018.json",
        },
        "had_2018_minival": {
            "data_dir": "test/positives",
            "ann_file": "annotations/had/instances_val2018.json",
        },
        "ark_2020_train": {
            "data_dir": "train",
            "ann_file": "annotations/ark/instances_train2020.json",
        },
        "ark_2020_val": {
            "data_dir": "test/positives",
            "ann_file": "annotations/ark/instances_val2020.json",
        },
        "ark_2020_minival": {
            "data_dir": "test/positives",
            "ann_file": "annotations/ark/instances_minival2020.json",
        },
        "blackvue_2020_train": {
            "data_dir": "train",
            "ann_file": "annotations/instances_train.json",
        },
        "blackvue_2020_val": {
            "data_dir": "test",
            "ann_file": "annotations/instances_val.json",
        },
    }

    @staticmethod
    def get(name, dataset_path):
        if "voc" in name:
            voc_root = DatasetCatalog.DATA_DIR
            if dataset_path != "":
                voc_root = dataset_path
            elif "VOC_ROOT" in os.environ:
                voc_root = os.environ["VOC_ROOT"]

            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                data_dir=os.path.join(voc_root, attrs["data_dir"]),
                split=attrs["split"],
            )
            return dict(factory="VOCDataset", args=args)
        elif (name in dataset for dataset in dataset_classes):
            coco_root = DatasetCatalog.DATA_DIR
            if dataset_path != "":
                coco_root = dataset_path
            elif "COCO_ROOT" in os.environ:
                coco_root = os.environ["COCO_ROOT"]

            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                data_dir=os.path.join(coco_root, attrs["data_dir"]),
                ann_file=os.path.join(coco_root, attrs["ann_file"]),
            )
            return dict(factory="COCODataset", args=args)

        raise RuntimeError("Dataset not available: {}".format(name))
