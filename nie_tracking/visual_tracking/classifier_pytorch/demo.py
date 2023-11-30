import argparse
from classifier import Classifier
from PIL import Image


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-net", type=str, required=True, help="net type")
    parser.add_argument(
        "-weights", type=str, required=True, help="the weights file you want to test"
    )
    parser.add_argument(
        "-input",
        type=str,
        required=True,
        help="the input image that you want to classify",
    )
    parser.add_argument(
        "-crop_box", type=float, help="Coordinates if image needs to be cropped"
    )
    parser.add_argument(
        "-print", action="store_false", help="Prints the image and the class"
    )
    parser.add_argument(
        "-dataset_type", type=str, help="specify the dataset type to get classes"
    )
    parser.add_argument(
        "-img_size", type=int, default=44, help="Image size for resizing"
    )

    args = parser.parse_args()
    img = Image.open(args.input)
    img.convert("RGB")

    classifier = Classifier(args.net, args.weights, args.dataset_type)
    classifier.infer_image(img, args.img_size, args.print)
