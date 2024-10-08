#!/usr/bin/env python

import sys

sys.path.insert(0, "../../")

import os
import torch
import torch.onnx
from torch.nn import BatchNorm2d

from encoding_custom.models import get_segmentation_model
import numpy as np
import argparse
import time
from encoding_custom.utils.energy_meter import EnergyMeter

parser = argparse.ArgumentParser(description="Get inputs.")
parser.add_argument(
    "--resume",
    type=str,
    help="Path to a directory to save the run log files and checkpoints in.",
    default=None,
)
parser.add_argument(
    "--dataset",
    type=str,
    help="Path to a directory to save the run log files and checkpoints in.",
    default="citys",
)
parser.add_argument(
    "--backbone",
    type=str,
    help="Path to a directory to save the run log files and checkpoints in.",
    default="resnet18",
)
parser.add_argument(
    "--model",
    type=str,
    help="Path to a directory to save the run log files and checkpoints in.",
    default="bisenet",
)
parser.add_argument(
    "--height",
    type=int,
    help="Path to a directory to save the run log files and checkpoints in.",
    default=512,
)
parser.add_argument(
    "--width",
    type=int,
    help="Path to a directory to save the run log files and checkpoints in.",
    default=512,
)
args = parser.parse_args()
# if args.resume is None:
#     args.resume = '/data/output/ahmed-badar/gather-excite-theta-minus-pretrain/runs/citys/bisenet/ge_theta_minus_pretrain_resnet18_ge_0016/model_best.pth.tar'
# args.dataset = 'citys'
# args.backbone = 'resnet18'
# args.model = 'bisenet'


def load_model(args):
    model = get_segmentation_model(
        args.model,
        dataset=args.dataset,
        backbone=args.backbone,
        aux=False,
        se_loss=False,
        norm_layer=BatchNorm2d,
        base_size=2048,
        crop_size=1024,
    )
    # Resuming checkpoint
    if args.resume is None or not os.path.isfile(args.resume):
        # raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
        print("no checkpoint found")
    else:
        checkpoint = torch.load(args.resume)
        # strict=False, so that it is compatible with old pytorch saved models
        if "state_dict" in checkpoint.keys():
            model.load_state_dict(checkpoint["state_dict"])
        elif "model" in checkpoint.keys():
            model.load_state_dict(checkpoint["model"])
        else:
            raise ("loaded checkpoint has no params key!")
        print(
            "=> loaded checkpoint '{}' (epoch {})".format(
                args.resume, checkpoint["epoch"]
            )
        )

    return model


def add_flops_counting_methods(net_main_module):
    """Adds flops counting functions to an existing model. After that
    the flops count should be activated and the model should be run on an input
    image.

    Example:

    fcn = add_flops_counting_methods(fcn)
    fcn = fcn.cuda().train()
    fcn.start_flops_count()

    _ = fcn(batch)

    fcn.compute_average_flops_cost() / 1e9 / 2 # Result in GFLOPs per image in batch

    Important: dividing by 2 only works for resnet models -- see below for the details
    of flops computation.

    Attention: we are counting multiply-add as two flops in this work, because in
    most resnet models convolutions are bias-free (BN layers act as bias there)
    and it makes sense to count muliply and add as separate flops therefore.
    This is why in the above example we divide by 2 in order to be consistent with
    most modern benchmarks. For example in "Spatially Adaptive Computatin Time for Residual
    Networks" by Figurnov et al multiply-add was counted as two flops.

    This module computes the average flops which is necessary for dynamic networks which
    have different number of executed layers. For static networks it is enough to run the network
    once and get statistics (above example).

    Implementation:
    The module works by adding batch_count to the main module which tracks the sum
    of all batch sizes that were run through the network.

    Also each convolutional layer of the network tracks the overall number of flops
    performed.

    The parameters are updated with the help of registered hook-functions which
    are being called each time the respective layer is executed.

    Parameters
    ----------
    net_main_module : torch.nn.Module
        Main module containing network

    Returns
    -------
    net_main_module : torch.nn.Module
        Updated main module with new methods/attributes that are used
        to compute flops.
    """

    # adding additional methods to the existing module object,
    # this is done this way so that each function has access to self object
    net_main_module.start_flops_count = start_flops_count.__get__(net_main_module)
    net_main_module.stop_flops_count = stop_flops_count.__get__(net_main_module)
    net_main_module.reset_flops_count = reset_flops_count.__get__(net_main_module)
    net_main_module.compute_average_flops_cost = compute_average_flops_cost.__get__(
        net_main_module
    )

    net_main_module.reset_flops_count()

    # Adding varialbles necessary for masked flops computation
    net_main_module.apply(add_flops_mask_variable_or_reset)

    return net_main_module


def compute_average_flops_cost(self):
    """
    A method that will be available after add_flops_counting_methods() is called
    on a desired net object.

    Returns current mean flops consumption per image.

    """

    batches_count = self.__batch_counter__

    flops_sum = 0

    for module in self.modules():

        if isinstance(module, torch.nn.Conv2d):

            flops_sum += module.__flops__

    return flops_sum / batches_count


def start_flops_count(self):
    """
    A method that will be available after add_flops_counting_methods() is called
    on a desired net object.

    Activates the computation of mean flops consumption per image.
    Call it before you run the network.

    """

    add_batch_counter_hook_function(self)

    self.apply(add_flops_counter_hook_function)


def stop_flops_count(self):
    """
    A method that will be available after add_flops_counting_methods() is called
    on a desired net object.

    Stops computing the mean flops consumption per image.
    Call whenever you want to pause the computation.

    """

    remove_batch_counter_hook_function(self)

    self.apply(remove_flops_counter_hook_function)


def reset_flops_count(self):
    """
    A method that will be available after add_flops_counting_methods() is called
    on a desired net object.

    Resets statistics computed so far.

    """

    add_batch_counter_variables_or_reset(self)

    self.apply(add_flops_counter_variable_or_reset)


def add_flops_mask(module, mask):
    def add_flops_mask_func(module):

        if isinstance(module, torch.nn.Conv2d):

            module.__mask__ = mask

    module.apply(add_flops_mask_func)


def remove_flops_mask(module):

    module.apply(add_flops_mask_variable_or_reset)


# ---- Internal functions


def conv_flops_counter_hook(conv_module, input, output):

    # Can have multiple inputs, getting the first one
    input = input[0]

    batch_size = input.shape[0]
    output_height, output_width = output.shape[2:]

    kernel_height, kernel_width = conv_module.kernel_size
    in_channels = conv_module.in_channels
    out_channels = conv_module.out_channels
    groups = conv_module.groups

    # We count multiply-add as 2 flops
    conv_per_position_flops = (
        2 * kernel_height * kernel_width * in_channels * out_channels / groups
    )

    active_elements_count = batch_size * output_height * output_width

    if conv_module.__mask__ is not None:

        # (b, 1, h, w)
        flops_mask = conv_module.__mask__.expand(
            batch_size, 1, output_height, output_width
        )
        active_elements_count = flops_mask.sum()

    overall_conv_flops = conv_per_position_flops * active_elements_count

    bias_flops = 0

    if conv_module.bias is not None:

        bias_flops = out_channels * active_elements_count

    overall_flops = overall_conv_flops + bias_flops

    conv_module.__flops__ += overall_flops


def batch_counter_hook(module, input, output):

    # Can have multiple inputs, getting the first one
    input = input[0]

    batch_size = input.shape[0]

    module.__batch_counter__ += batch_size


def add_batch_counter_variables_or_reset(module):

    module.__batch_counter__ = 0


def add_batch_counter_hook_function(module):

    if hasattr(module, "__batch_counter_handle__"):

        return

    handle = module.register_forward_hook(batch_counter_hook)
    module.__batch_counter_handle__ = handle


def remove_batch_counter_hook_function(module):

    if hasattr(module, "__batch_counter_handle__"):

        module.__batch_counter_handle__.remove()

        del module.__batch_counter_handle__


def add_flops_counter_variable_or_reset(module):

    if isinstance(module, torch.nn.Conv2d):

        module.__flops__ = 0


def add_flops_counter_hook_function(module):

    if isinstance(module, torch.nn.Conv2d):

        if hasattr(module, "__flops_handle__"):

            return

        handle = module.register_forward_hook(conv_flops_counter_hook)
        module.__flops_handle__ = handle


def remove_flops_counter_hook_function(module):

    if isinstance(module, torch.nn.Conv2d):

        if hasattr(module, "__flops_handle__"):

            module.__flops_handle__.remove()

            del module.__flops_handle__


# --- Masked flops counting


# Also being run in the initialization
def add_flops_mask_variable_or_reset(module):

    if isinstance(module, torch.nn.Conv2d):

        module.__mask__ = None


def main():
    net = load_model(args)
    add_flops_counting_methods(net)
    net = net.cuda()
    net.start_flops_count()
    net.eval()
    run_time = list()
    input = torch.randn(1, 3, args.height, args.width).cuda()
    # ensure that context initialization and normal_() operations
    # finish before you start measuring time
    torch.cuda.synchronize()
    torch.cuda.synchronize()

    # single inference for warm-up
    with torch.no_grad():
        output = net(input)
    torch.cuda.synchronize()  # wait for mm to finish

    N = 10000
    run_time = [0] * N

    with EnergyMeter() as em:
        for i in range(0, N):
            start = time.perf_counter()
            with torch.no_grad():
                output = net(input)
            torch.cuda.synchronize()  # wait for mm to finish
            run_time[i] = time.perf_counter() - start
        torch.cuda.synchronize()
        print(f"Energy per image:  {em.energy/N:.2} J")

    print(f"Processing time per image: {np.mean(run_time)*1000:.3} ms")
    print(f"Mean fps is:               {(1.0 / np.mean(run_time))}")
    print(
        f"FLOPs per image:           {(net.compute_average_flops_cost() / 1e9 / 2)} GFLOPs"
    )


if __name__ == "__main__":
    main()
