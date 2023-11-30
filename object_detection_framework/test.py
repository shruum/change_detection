#!/usr/bin/env python3
import argparse
import logging
import os
import yaml

import torch
import torch.utils.data

from od.engine.inference import do_evaluation
from od.modeling.detector import build_detection_model
from od.utils import dist_util
from od.utils.checkpoint import CheckPointer
from od.utils.dist_util import synchronize
from od.utils.logger import setup_logger

from od.default_config import cfg
from od.default_config import centernet_cfg
from od.utils.flops_counter import get_model_complexity_info
from od.utils.energy_meter import EnergyMeter
from contextlib import ExitStack

sub_cfg_dict = {"CenterNetHead": centernet_cfg}


def evaluation(cfg, ckpt, eval_only, calc_energy, distributed):
    logger = logging.getLogger("Object Detection.inference")

    model = build_detection_model(cfg)

    checkpointer = CheckPointer(model, save_dir=cfg.OUTPUT_DIR, logger=logger)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)
    checkpointer.load(ckpt, use_latest=ckpt is None)

    if not eval_only:
        image_res = (3, cfg.INPUT.IMAGE_SIZE, cfg.INPUT.IMAGE_SIZE)
        flops_count, params_count = get_model_complexity_info(model, image_res)
        print("MAC Count:", flops_count)
        print("Number of Parameters:", params_count)

    with EnergyMeter(dir=cfg.OUTPUT_DIR) if calc_energy else ExitStack():
        do_evaluation(
            cfg, model, distributed, get_inf_time=False if eval_only else True
        )


def main():
    parser = argparse.ArgumentParser(description="Evaluation on VOC and COCO dataset.")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--ckpt",
        help="The path to the checkpoint for test, default is the latest checkpoint.",
        default=None,
        type=str,
    )

    parser.add_argument(
        "--output_dir",
        default="eval_results",
        type=str,
        help="The directory to store evaluation results.",
    )

    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="If set, outputs MAP without other statistics such as Inference time, Energy and Number of Parameters",
    )
    parser.add_argument(
        "--calc_energy",
        action="store_true",
        help="If set, measures and outputs the Energy consumption",
    )
    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed = num_gpus > 1

    if torch.cuda.is_available():
        # This flag allows you to enable the inbuilt cudnn auto-tuner to
        # find the best algorithm to use for your hardware.
        torch.backends.cudnn.benchmark = True
    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    head = yaml.load(open(args.config_file), Loader=yaml.FullLoader)["MODEL"]["HEAD"][
        "NAME"
    ]
    sub_cfg = sub_cfg_dict[head]

    cfg.merge_from_other_cfg(sub_cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    logger = setup_logger("Object Detection", dist_util.get_rank(), cfg.OUTPUT_DIR)
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))
    evaluation(
        cfg,
        ckpt=args.ckpt,
        eval_only=args.eval_only,
        calc_energy=args.calc_energy,
        distributed=distributed,
    )


if __name__ == "__main__":
    main()
