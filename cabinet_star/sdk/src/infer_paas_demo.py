#!/usr/bin/env python3
# This is a very simplistic way how to use the inference SDK.
# This example creates an instance of the Engine class with the default values, but it requires the result to be in
# JSON format and it doesn't want to save the result to disk. This can be the use case of running inference on a
# flask server.
#
# The checkpoint is pick up automatically from the model folder.
#
# The eng.infer(input) method could be called multiple times with different inputs as long as the eng object exists.
#

import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../.."))
from sdk.src.engine import Engine


def main():
    eng = Engine(output_dir="output/", output_write=True, output_format="json",)
    eng.load_model()
    result = eng.infer("/workspace/sample_data/image0.jpg")
    print(result)


if __name__ == "__main__":
    main()
