from glob import glob
from segmentation import Segmenter
from lanes.vpoint_debug import main


if __name__ == "__main__":
    # imports only needed for debugging/testing

    segmenter = Segmenter("/tmp")
    # img_paths = ["./lanes/essence/0009.jpg", "./lanes/essence/0377.jpg"]
    # img_paths = [ "./lanes/essence/DE_20190701-121947-NF_00000078_7.07697780_51.46475458.jpg" ]
    img_paths = glob("./sample_images/lanes/*.jpg")
    main(img_paths, segmenter)
    # vpoints = localize_visualize_vanishing_point(img_paths, segmenter)
