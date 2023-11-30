# Copyright (C) 2020 by NavInfo Europe B.V. The Netherlands - All rights reserved
# Information classification: Confidential
# This content is protected by international copyright laws.
# Reproduction and distribution is prohibited without written permission.
import argparse
import os


def dir_path(string):
    if os.path.isdir(string) and os.path.exists(string):
        return string
    else:
        raise NotADirectoryError(string)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Change Rate Detection",
        # formatter_class=argparse.RawTextHelpFormatter,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--run1",
        type=str,
        help="""Input chinese info txt file for run1,
                for blackvue format this is the directory containing images.""",
    )

    parser.add_argument(
        "--run2",
        type=str,
        help="""Input chinese info txt file for run2,
                for blackvue format this is the directory containing images.""",
    )

    parser.add_argument(
        "-w",
        "--write_images",
        help="Write images along with alignment at cache_dir",
        default=False,
        action="store_true",
    )

    parser.add_argument(
        "-c",
        "--clear_cache",
        help="Delete the cache directory",
        default=False,
        action="store_true",
    )

    parser.add_argument(
        "--cache_dir",
        default="./cache/",
        help="Directory to store intermediate results",
        type=str,
    )

    parser.add_argument(
        "-nsc",
        "--no_speed_correction",
        help="A much faster but inaccurate/unstable alignment! Use with care",
        default=False,
        action="store_true",
    )

    parser.add_argument(
        "-f",
        "--format",
        default="default",
        choices=["default", "blackvue_images", "blackvue_video", "aiim", "fangzhou"],
        help="If input data is in AIIM data format, run_1 and run_2 should be the directories and subdriectories with images ",
    )

    parser.add_argument(
        "-q",
        "--quiet",
        help="Set logging level to WARNING (default INFO)",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        help="Set logging level to DEBUG (default INFO)",
        default=False,
        action="store_true",
    )
    args = parser.parse_args()

    # CHECKS & PROCESSING

    assert (
        args.quiet == False or args.verbose == False
    ), "--quiet and -verbose are mutually exculsive"

    if args.run1 and args.run2:
        dataset_name = find_dataset_name(args.run1, args.run2)
        args.cache_dir = os.path.join(args.cache_dir, dataset_name)
        os.makedirs(args.cache_dir, exist_ok=True)

    # PRINTING

    print("\nargument values:")
    for arg in sorted(vars(args)):
        print("{}:".format(arg).ljust(25), getattr(args, arg))
    print("")

    return args


def find_dataset_name(str1, str2):
    """kind of finding the largest common substring"""
    dataset_name = ""
    word = ""
    for c1, c2 in zip(str1, str2):
        if c1 != c2:
            break
        if c1 == "/":
            dataset_name = dataset_name + word + "_"
            word = ""
        else:
            word = word + c1
    return dataset_name
