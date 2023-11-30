# Conda environment

Create the conda environment:

    conda env create -f environment.yml

Activate environment

    conda activate nie_tracking

# Get dependencies

Get submodules:

    git submodule update --init --recursive

Install dependencies if necessary:

    apt-get install -y ninja-build
    apt-get install libturbojpeg

Run build script:

    ./build.sh

# Test tracking

Set a sensible output path:

    OUTPUTPATH=/volumes1/terence.brouns/Projects/ChangeRateDetection/output/

Run tracking:

    python tracker.py --input_path="/data/output/ratnajit/CRDoutputdir/images/run1/" --detection_path="/data/output/ratnajit/CRDoutputdir/detections/run1/" --output_path=$OUTPUTPATH --upper_conf_threshold=0.5 --input_type="image" --run_classifier --classifier_model="/data/output/ratnajit/CRDoutputdir/classifier_model/resnet34-140-best.pth" --save_crop_images