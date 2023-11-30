import logging
import os

import torch
import torch.utils.data
from tqdm import tqdm
import time
import numpy as np

from od.data.build import make_data_loader
from od.data.datasets.evaluation import evaluate

from od.utils import dist_util, mkdir
from od.utils.dist_util import synchronize, is_main_process


def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu):
    all_predictions = dist_util.all_gather(predictions_per_gpu)
    if not dist_util.is_main_process():
        return
    # merge the list of dicts
    predictions = {}
    for p in all_predictions:
        predictions.update(p)
    # convert a dict where the key is the index in a list
    image_ids = list(sorted(predictions.keys()))
    if len(image_ids) != image_ids[-1] + 1:
        logger = logging.getLogger("Object Detection.inference")
        logger.warning(
            "Number of images that were gathered from multiple processes is not "
            "a contiguous set. Some images might be missing from the evaluation"
        )

    # convert to a list
    predictions = [predictions[i] for i in image_ids]
    return predictions


def compute_on_dataset(model, data_loader, device):
    results_dict = {}
    cpu_device = torch.device("cpu")
    for batch in tqdm(data_loader):
        images, targets, image_ids = batch
        with torch.no_grad():
            outputs = model(images.to(device))

            outputs = [o.to(cpu_device) for o in outputs]
        results_dict.update(
            {img_id: result for img_id, result in zip(image_ids, outputs)}
        )
    return results_dict


def get_inference_time(model, data_loader, device, test_it=501):
    run_time = []
    data_sampler = iter(data_loader)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    with torch.no_grad():
        for i in range(test_it):
            try:
                images, targets, image_ids = next(data_sampler)
            except StopIteration:
                batch_iterator = iter(data_loader)
                images, targets, image_ids = next(batch_iterator)
            images = images.to(device)
            torch.cuda.synchronize()

            start.record()
            outputs = model(images)
            end.record()

            torch.cuda.synchronize()
            run_time.append(start.elapsed_time(end))

    run_time = run_time[1:]
    avg_run_time = np.mean(run_time)

    return avg_run_time


def inference(
    cfg,
    model,
    data_loader,
    dataset_name,
    device,
    get_inf_time=False,
    output_folder=None,
    use_cached=False,
    **kwargs
):
    dataset = data_loader.dataset
    logger = logging.getLogger("Object Detection.inference")
    logger.info("Evaluating {} dataset({} images):".format(dataset_name, len(dataset)))
    predictions_path = os.path.join(output_folder, "predictions.pth")

    if get_inf_time:
        inf_time = get_inference_time(model, data_loader, device)
        print("Model Inference Time:", inf_time, "ms")

    if use_cached and os.path.exists(predictions_path):
        predictions = torch.load(predictions_path, map_location="cpu")
    else:
        predictions = compute_on_dataset(model, data_loader, device)
        synchronize()
        predictions = _accumulate_predictions_from_multiple_gpus(predictions)
    if not is_main_process():
        return
    if output_folder:
        torch.save(predictions, predictions_path)
    return evaluate(
        cfg,
        dataset=dataset,
        predictions=predictions,
        output_dir=output_folder,
        **kwargs
    )


@torch.no_grad()
def do_evaluation(cfg, model, distributed, get_inf_time=False, **kwargs):
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module
    model.eval()
    device = torch.device(cfg.MODEL.DEVICE)
    data_loaders_val = make_data_loader(cfg, is_train=False, distributed=distributed)
    eval_results = []
    for dataset_name, data_loader in zip(cfg.DATASETS.TEST, data_loaders_val):
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
        if not os.path.exists(output_folder):
            mkdir(output_folder)
        eval_result = inference(
            cfg,
            model,
            data_loader,
            dataset_name,
            device,
            get_inf_time,
            output_folder,
            **kwargs
        )
        eval_results.append(eval_result)
    return eval_results
