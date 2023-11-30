#!/usr/bin/env python3
# This is a very simplistic way how to use the inference SDK.
# This example creates an instance of the Engine class with the default values, but it requires the result to be in
# JSON format and it doesn't want to save the result to disk. This can be the use case of running inference on a
# flask server.
#
# The checkpoint and the configuration files are pick up automatically from the model folder which needs to be
# preconfigured with the correct configuration and checkpoint files.
#
# The eng.infer(input) method could be called multiple times with different inputs as long as the eng object exists.
#
# Note: This example is for the VOC source code. The Engine class needs to be instantiated with dataset_type='coco'
# in order to run on the COCO source code.

import os
import sys

HOME_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")
sys.path.append(HOME_DIR)
# pylint: disable=wrong-import-position
from sdk.src.engine import Engine

# pylint: enable=wrong-import-position


def main():
    eng = Engine(output_format="json", output_write=False)
    eng.load_model()
    result = eng.infer(os.path.join(HOME_DIR, "demo/0001.jpg"))
    print(result)


if __name__ == "__main__":
    main()
