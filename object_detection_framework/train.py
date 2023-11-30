import argparse
import logging
import os
import yaml

import torch
import torch.distributed as dist
from numpy import random
from od.engine.inference import do_evaluation
from od.data.build import make_data_loader
from od.engine.trainer import do_train
from od.modeling.detector import build_detection_model
from od.solver import make_optimizer
from od.solver.lr_scheduler import make_lr_scheduler
from od.utils import dist_util, mkdir
from od.utils.checkpoint import CheckPointer
from od.utils.dist_util import synchronize
from od.utils.logger import setup_logger
from od.utils.misc import str2bool

from od.default_config import cfg
from od.default_config import centernet_cfg

sub_cfg_dict = {"CenterNetHead": centernet_cfg}


def train(cfg, args):
    logger = logging.getLogger(cfg.LOGGER.NAME + ".trainer")
    model = build_detection_model(cfg)

    if args.distributed:
        torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        device = torch.device("cuda:{}".format(args.local_rank))
        model.to(device)
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True,
        )
    else:
        device = torch.device(cfg.MODEL.DEVICE)
        model.to(device)

    lr = cfg.SOLVER.LR * args.num_gpus  # scale by num gpus
    optimizer = make_optimizer(cfg, model, lr)

    milestones = [step // args.num_gpus for step in cfg.SOLVER.LR_STEPS]
    scheduler = make_lr_scheduler(cfg, optimizer, milestones)

    arguments = {"iteration": 0}
    save_to_disk = dist_util.get_rank() == 0
    checkpointer = CheckPointer(
        model, optimizer, scheduler, cfg.OUTPUT_DIR, save_to_disk, logger
    )
    extra_checkpoint_data = checkpointer.load()
    arguments.update(extra_checkpoint_data)

    max_iter = cfg.SOLVER.MAX_ITER // args.num_gpus
    train_loader = make_data_loader(
        cfg,
        is_train=True,
        distributed=args.distributed,
        max_iter=max_iter,
        start_iter=arguments["iteration"],
    )

    model = do_train(
        cfg,
        model,
        train_loader,
        optimizer,
        scheduler,
        checkpointer,
        device,
        arguments,
        args,
    )
    return model


def main():
    # arguments
    # any New config should be added  to config file and you  pass this config file at the arguments
    parser = argparse.ArgumentParser(
        description="Single Shot MultiBox Detector Training With PyTorch"
    )
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--log_step", default=10, type=int, help="Print logs every log_step"
    )
    parser.add_argument(
        "--save_step", default=2500, type=int, help="Save checkpoint every save_step"
    )
    parser.add_argument(
        "--eval_step",
        default=2500,
        type=int,
        help="Evaluate dataset every eval_step, disabled when eval_step < 0",
    )
    parser.add_argument("--use_tensorboard", default=True, type=str2bool)
    parser.add_argument(
        "--seed",
        default=0,
        type=int,
        help="Random seed for processes. Seed must be fixed for distributed training",
    )
    parser.add_argument(
        "--skip-test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true",
    )
    parser.add_argument(
        "--calc_energy",
        action="store_true",
        help="If set, measures and outputs the Energy consumption",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()

    num_gpus = torch.cuda.device_count()
    args.distributed = num_gpus > 1
    args.num_gpus = num_gpus
    # remove torch.backends.cudnn.benchmark to avoid potential risk

    if args.distributed:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=num_gpus,
            rank=args.local_rank,
        )
        synchronize()

    # defined by 'head' not meta_architecture
    head = yaml.load(open(args.config_file), Loader=yaml.FullLoader)["MODEL"]["HEAD"][
        "NAME"
    ]
    sub_cfg = sub_cfg_dict[head]

    cfg.merge_from_other_cfg(sub_cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    if cfg.OUTPUT_DIR:
        mkdir(cfg.OUTPUT_DIR)

    logger = setup_logger(cfg.LOGGER.NAME, dist_util.get_rank(), cfg.OUTPUT_DIR)
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    model = train(cfg, args)

    if not args.skip_test:
        logger.info("Start evaluating...")
        torch.cuda.empty_cache()  # speed up evaluating after training finished
        do_evaluation(cfg, model, distributed=args.distributed)


if __name__ == "__main__":
    main()
