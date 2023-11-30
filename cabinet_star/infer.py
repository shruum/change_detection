#!/usr/bin/env python3

import argparse

from sdk.src.engine import *


def main():
    parser = argparse.ArgumentParser(description="Inference application.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Model name")
    parser.add_argument(
        "--backbone", type=str, default=DEFAULT_BACKBONE, help="Backbone name"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=DEFAULT_CKPT,
        help="Trained model in .pth.tar format",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=DEFAULT_INPUT_DIR,
        help="Input directory/image to be predicted (default: %(default)s).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory that will contain the prediction(s) (default: %(default)s).",
    )
    parser.add_argument(
        "--output_format",
        type=str,
        choices=["img", "json"],
        default=DEFAULT_OUTPUT_FORMAT,
        help="Output format (default: %(default)s).",
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose mode.")
    args = parser.parse_args()
    if args.verbose:
        print(args)

    eng = Engine(
        model=args.model,
        backbone=args.backbone,
        ckpt=args.checkpoint,
        output_dir=args.output_dir,
        output_format=args.output_format,
        output_write=True,
        verbose=args.verbose,
    )
    eng.load_model()
    eng.infer(args.input)


if __name__ == "__main__":
    main()
