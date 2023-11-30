from torch.utils.data import ConcatDataset

from od.default_config.path_catlog import DatasetCatalog
from .voc import VOCDataset
from .coco import COCODataset

_DATASETS = {
    "VOCDataset": VOCDataset,
    "COCODataset": COCODataset,
}


def build_dataset(
    dataset_list, cfg, transform=None, target_transform=None, is_train=True
):
    assert len(dataset_list) > 0
    datasets = []
    dataset_path = cfg.DATASETS.PATH
    for dataset_name in dataset_list:
        data = DatasetCatalog.get(dataset_name, dataset_path)
        args = data["args"]
        factory = _DATASETS[data["factory"]]
        args["include_background"] = cfg.DATA_LOADER.INCLUDE_BACKGROUND
        args["transform"] = transform
        args["target_transform"] = target_transform
        if factory == VOCDataset:
            args["keep_difficult"] = not is_train
        elif factory == COCODataset:
            args["remove_empty"] = is_train
        dataset = factory(**args)
        datasets.append(dataset)
    # for testing, return a list of datasets
    if not is_train:
        return datasets
    dataset = datasets[0]
    if len(datasets) > 1:
        dataset = ConcatDataset(datasets)

    return [dataset]
