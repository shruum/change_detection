#!/bin/bash

echo "***Building external libraries for detection***"
cd object_detection_framework/ext && python3 build.py build_ext develop

echo "***Building Deformable Convolution library***"
cd ../od/modeling/head/centernet/DCNv2 && ./make.sh

echo "***Setup pytracking***"
cd ../../../../../../nie_tracking/visual_tracking/pytracking/

python3 -c "from pytracking.evaluation.environment import create_default_local_file; create_default_local_file()"
python3 -c "from ltr.admin.environment import create_default_local_file; create_default_local_file()"
mkdir -p pytracking/networks
cp /data/nie/teams/arl/models_checkpoints/crd/2.0/traffic_signs/tracker/* pytracking/networks/
cd ../