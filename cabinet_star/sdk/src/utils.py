import pathlib
import os


def get_mean_std(model_name, dataset_type):
    use_mapillary_norms = False
    if model_name == "rgpnet" and dataset_type == "mapillary_merged":
        use_mapillary_norms = True

    if use_mapillary_norms:
        mean = [0.41738699, 0.45732192, 0.46886091]
        std = [0.25685097, 0.26509955, 0.29067996]
    else:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

    return mean, std


def find_files_with_extensions(where, extensions, recursive=False):
    """Return all the files from `where` which match one of the `extensions`."""
    image_paths = []
    for ext in extensions:
        if recursive:
            image_paths += sorted(pathlib.Path(where).rglob("*" + ext))
        else:
            image_paths += sorted(pathlib.Path(where).glob("*" + ext))
    return [str(path.absolute()) for path in image_paths]
