# Object Detection as a Service

Documentation for Object Detection as a Service

- [Object Detection as a Service](#object-detection-as-a-service)
    - [1. Environment Preparation](#1-environment-preparation)
        - [a) First a conda environment is needed for the repo: (Note: if you want to skip conda, you can skip this step)](#a-first-a-conda-environment-is-needed-for-the-repo-note-if-you-want-to-skip-conda-you-can-skip-this-step)
          - [Following conda environment creation, we need to activate the environment](#following-conda-environment-creation-we-need-to-activate-the-environment)
          - [The shell prompt should look like:](#the-shell-prompt-should-look-like)
        - [b) Install the necessary packages:](#b-install-the-necessary-packages)
        - [c) Clone the Repository from NavInfo Bitbucket (either HTTPS or SSH):](#c-clone-the-repository-from-navinfo-bitbucket-either-https-or-ssh)
        - [d) Build the external dependencies (additional requirements)](#d-build-the-external-dependencies-additional-requirements)
    - [2. Setting up datasets](#2-setting-up-datasets)
        - [a) PASCAL VOC](#a-pascal-voc)
          - [The same dataset can also be found in the file server under the location:](#the-same-dataset-can-also-be-found-in-the-file-server-under-the-location)
        - [b) Microsoft COCO](#b-microsoft-coco)
          - [Unzip the downloaded zips and put them in a folder whose structure is as follows:](#unzip-the-downloaded-zips-and-put-them-in-a-folder-whose-structure-is-as-follows)
          - [Similarly, the same dataset can also be found in the file server under the location:](#similarly-the-same-dataset-can-also-be-found-in-the-file-server-under-the-location)
          - [Finally, the bash environment needs to be pointed to the correct dataset (to be used for training).](#finally-the-bash-environment-needs-to-be-pointed-to-the-correct-dataset-to-be-used-for-training)
    - [3. Train OD as a Service](#3-train-od-as-a-service)
        - [a) Choosing the correct (backbone and head)](#a-choosing-the-correct-backbone-and-head)
        - [b) Training SSD head with a specific backbone:](#b-training-ssd-head-with-a-specific-backbone)
          - [You will see an output such as:](#you-will-see-an-output-such-as)
        - [c) Train backbones with CenterNet Head](#c-train-backbones-with-centernet-head)
    - [4. Evaluate OD as a Service](#4-evaluate-od-as-a-service)
          - [Once the evaluation process completes, the output should look like this:](#once-the-evaluation-process-completes-the-output-should-look-like-this)
    - [5. Inference with OD as a Service:](#5-inference-with-od-as-a-service)
          - [The output of the process should look like:](#the-output-of-the-process-should-look-like)

### 1. Environment Preparation

##### a) First a conda environment is needed for the repo: (Note: if you want to skip conda, you can skip this step)
```
conda create -n pytorch_odservice python=3.7.4 anaconda
```
###### Following conda environment creation, we need to activate the environment
```
conda activate pytorch_odservice
```
###### The shell prompt should look like:
```
(pytorch_odservice) ratnajit.mukherjee@nides038:/....
```
##### b) Install the necessary packages:
```
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
pip install numpy yacs tqdm opencv-python vizer tensorboardx pycocotools pynvml
```

For all the correct package versions, you can consult `requirements_virtualenv.txt`.
Alternatively, you can prepare the environment using:
```
pip install -r requirements.txt
```

##### c) Clone the Repository from NavInfo Bitbucket (either HTTPS or SSH):
```
https://bitbucket.navinfo.eu/scm/evvis/object_detection_framework.git
ssh://git@navbitbucket01.navinfo.eu:7999/evvis/object_detection_framework.git
```
Once the repository in downloaded in your preferred folder, navigate to
that root folder.

##### d) Build the external dependencies (additional requirements)

```Bash
# For faster inference you need to build nms, this is needed during evaluating.
(cd ext && python build.py build_ext develop)

# For ThunderNet, extra compilation is needed before running.
(cd od/modeling/head/ThunderNet && python setup.py build develop)

# For CenterNet, extra compilation is needed before running.
[ -d od/modeling/head/centernet/DCNv2 ] && (cd od/modeling/head/centernet/DCNv2 && ./make.sh)
```

### 2. Setting up datasets
##### a) PASCAL VOC
The VOC datasets can be downloaded into local machines or can be used from the data server.

For local machines download and extract:
```
# PASCAL VOC 2007
http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
tar -xvf VOCtrainval_06-Nov-2007.tar
tar -xvf VOCtest_06-Nov-2007.tar

# PASCAL VOC 2012
http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
tar -xvf VOCtrainval_11-May-2012.tar
```
This process will produce a directory called **VOCdevkit** (in your chosen path)
where the folder structure is as follows:
```
VOCdevkit
|__ VOC2007
    |_ JPEGImages
    |_ Annotations
    |_ ImageSets
    |_ SegmentationClass
|__ VOC2012
    |_ JPEGImages
    |_ Annotations
    |_ ImageSets
    |_ SegmentationClass
|__ ...
```
###### The same dataset can also be found in the file server under the location:
```
/data/input/datasets/VOCdevkit
```
##### b) Microsoft COCO
Similar to VOC, the COCO dataset can be downloaded and extracted in local machines
in the following way:
```
# Images:
http://images.cocodataset.org/zips/train2014.zip
http://images.cocodataset.org/zips/val2014.zip
http://images.cocodataset.org/zips/test2014.zip

# Annotations:
http://images.cocodataset.org/annotations/annotations_trainval2014.zip
```
###### Unzip the downloaded zips and put them in a folder whose structure is as follows:
**Note: If you download and extract one file will be missing i.e. "instances_valminusminival2014.json"
which can be found in the server (check the paragraph below). The file can be copied to the local machine if desired.**
```
COCO
|__ annotations
    |_ instances_valminusminival2014.json
    |_ instances_minival2014.json
    |_ instances_train2014.json
    |_ instances_val2014.json
    |_ ...
|__ train2014
    |_ <im-1-name>.jpg
    |_ ...
    |_ <im-N-name>.jpg
|__ val2014
    |_ <im-1-name>.jpg
    |_ ...
    |_ <im-N-name>.jpg
|__ ...
```
###### Similarly, the same dataset can also be found in the file server under the location:
```
/data/input/datasets/COCO
```
###### Finally, the bash environment needs to be pointed to the correct dataset (to be used for training).
```
# for VOC
export VOC_ROOT="<path to VOCdevkit directory"
# for COCO
export COCO_ROOT="<path to COCO directory"
```
### 3. Train OD as a Service
Object detection service consists of multiple backbones and heads wherein
each combination (backbone and head) can be used to train an detection model.

The complete list of combinations is given in:
```
https://confluence.navinfo.eu/display/AIDET/Object+Detection+as+a+Service
```
##### a) Choosing the correct (backbone and head)
The combination for backbone and head can be chosen using a configuration
file. The configuration (.yaml) files can be found:
```
# root folder for the config files
cd configs

# for SSD head
cd configs/ssd

# for CenterNet Head
cd configs/centernet

# for yolo head
cd configs/yolo
...

# To train a backbone and head (you can create a new configuration file)
with the name
<backbone_name>_<headname><imagesize>_<dataset>

For example:
resnet18_ssd512_voc0712.yaml (Backbone: Resnet18, Head: ssd, Image size:512, Dataset: voc0712)
```
##### b) Training SSD head with a specific backbone:
Navigate back to the root folder:
```
# Example 1: Resnet18 backbone and SSD head:
python train.py --config-file configs/ssd/resnet18_ssd512_voc0712.yaml

# Example 2: PeleeNet backbone and SSD head:
python train.py --config-file configs/ssd/peleenet_ssd512_voc0712.yaml
```
###### You will see an output such as:
```
2019-10-07 13:43:50,886 SSD.trainer INFO: Using 1 GPUs
2019-10-07 13:43:50,886 SSD.trainer INFO: Namespace(config_file='configs/ssd/resnet18_ssd512_voc0712.yaml', distributed=False, eval_step=2500, local_rank=0, log_step=10, num_gpus=1, opts=[], save_step=2500, skip_test=False, use_tensorboard=True)
2019-10-07 13:43:50,886 SSD.trainer INFO: Loaded configuration file configs/ssd/resnet18_ssd512_voc0712.yaml
2019-10-07 13:43:50,886 SSD.trainer INFO:
MODEL:
  HEAD:
    NAME: 'SSDBoxHead'
  NUM_CLASSES: 21
  BACKBONE:
    NAME: 'resnet18'
    OUT_CHANNELS: (128,512,512,512,512,512)
  PRIORS:
    FEATURE_MAPS: [64, 32, 16, 8, 4, 2]
    STRIDES: [8, 16, 32, 64, 128, 256]
    MIN_SIZES: [35.84, 76.8, 153.6, 230.4, 307.2, 384.0]
    MAX_SIZES: [76.8, 153.6, 230.4, 307.2, 384.0, 460.8]
    ASPECT_RATIOS: [[2], [2, 3], [2, 3], [2, 3], [2, 3], [2]]
    BOXES_PER_LOCATION: [4, 6, 6, 6, 6, 4]
INPUT:
  IMAGE_SIZE: 512
DATASETS:
  TRAIN: ("voc_2007_trainval", "voc_2012_trainval")
  TEST: ("voc_2007_test", )
SOLVER:
  MAX_ITER: 120000
  LR_STEPS: [80000, 100000]
  GAMMA: 0.1
  BATCH_SIZE: 32
  LR: 1e-3

OUTPUT_DIR: 'outputs/resnet18_ssd512_voc0712'
2019-10-07 13:43:50,887 SSD.trainer INFO: Running with config:
DATASETS:
  TEST: ('voc_2007_test',)
  TRAIN: ('voc_2007_trainval', 'voc_2012_trainval')
DATA_LOADER:
  NUM_WORKERS: 8
  PIN_MEMORY: True
INPUT:
  IMAGE_SIZE: 512
  PIXEL_MEAN: [123, 117, 104]
LOGGER:
  NAME: SSD.trainer
MODEL:
  BACKBONE:
    NAME: resnet18
    OUT_CHANNELS: (128, 512, 512, 512, 512, 512)
    PRETRAINED: True
    WIDTH_MULT: 1.0
  BOX_PREDICTOR: SSDBoxPredictor
  CENTER_VARIANCE: 0.1
  DEVICE: cuda
  HEAD:
    NAME: SSDBoxHead
  META_ARCHITECTURE: TwoPhasesDetector
  NEG_POS_RATIO: 3
  NUM_CLASSES: 21
  PRIORS:
    ASPECT_RATIOS: [[2], [2, 3], [2, 3], [2, 3], [2, 3], [2]]
    BOXES_PER_LOCATION: [4, 6, 6, 6, 6, 4]
    CLIP: True
    FEATURE_MAPS: [64, 32, 16, 8, 4, 2]
    MAX_SIZES: [76.8, 153.6, 230.4, 307.2, 384.0, 460.8]
    MIN_SIZES: [35.84, 76.8, 153.6, 230.4, 307.2, 384.0]
    STRIDES: [8, 16, 32, 64, 128, 256]
  SIZE_VARIANCE: 0.2
  THRESHOLD: 0.5
OUTPUT_DIR: outputs/resnet18_ssd512_voc0712
SOLVER:
  BATCH_SIZE: 32
  GAMMA: 0.1
  LR: 0.001
  LR_STEPS: [80000, 100000]
  MAX_ITER: 120000
  MOMENTUM: 0.9
  NAME: SGD_optimizer
  WARMUP_FACTOR: 0.3333333333333333
  WARMUP_ITERS: 500
  WEIGHT_DECAY: 0.0005
TEST:
  BATCH_SIZE: 10
  CONFIDENCE_THRESHOLD: 0.01
  MAX_PER_CLASS: -1
  MAX_PER_IMAGE: 100
  NMS_THRESHOLD: 0.45
2019-10-07 13:43:52,471 SSD.trainer.trainer INFO: No checkpoint found.
2019-10-07 13:43:52,486 SSD.trainer INFO: Start training ...
2019-10-07 13:44:12,570 SSD.trainer INFO: iter: 000010, lr: 0.00035, total_loss: 29.709 (29.709), reg_loss: 5.241 (5.241), cls_loss: 24.467 (24.467), time: 2.003 (2.003), eta: 2 days, 18:45:28, mem: 8487M
2019-10-07 13:44:19,998 SSD.trainer INFO: iter: 000020, lr: 0.00036, total_loss: 19.760 (24.734), reg_loss: 3.638 (4.440), cls_loss: 16.121 (20.294), time: 0.743 (1.373), eta: 1 day, 21:45:14, mem: 8487M
2019-10-07 13:44:27,454 SSD.trainer INFO: iter: 000030, lr: 0.00037, total_loss: 15.654 (21.707), reg_loss: 3.121 (4.000), cls_loss: 12.533 (17.707), time: 0.746 (1.164), eta: 1 day, 14:46:58, mem: 8487M
2019-10-07 13:44:34,903 SSD.trainer INFO: iter: 000040, lr: 0.00039, total_loss: 13.818 (19.735), reg_loss: 2.949 (3.738), cls_loss: 10.868 (15.997), time: 0.745 (1.059), eta: 1 day, 11:17:22, mem: 8487M
2019-10-07 13:44:42,347 SSD.trainer INFO: iter: 000050, lr: 0.00040, total_loss: 12.594 (18.307), reg_loss: 2.973 (3.585), cls_loss: 9.621 (14.722), time: 0.744 (0.996), eta: 1 day, 9:11:24, mem: 8487M
...
```
##### c) Train backbones with CenterNet Head
As of now, the CenterNet head is available in a separate branch. Switch to:
```
git checkout AIDET-267-CenterNet-Head-WC
```
Navigate to the configs directory. If CenterNetHead directory exists, then copy the config files from:
```
/data/output/ratnajit/CenterNet/config_files
```
else you have to create a config files which should be like:
```
MODEL:
  NUM_CLASSES: 20
  BACKBONE:
    NAME: 'resnet18'
    OUT_CHANNEL: 512
  HEAD:
    NAME: 'CenterNetHead'
    BACKBONE_FEATURE: 16
    HEAD_CONFIG: {'hm':20, 'wh': 2, 'reg': 2}
    LOSS_WEIGHTS: {'hm':1, 'wh': 0.1, 'reg': 1}
    NUM_DECONV_LAYERS: 3
    DECONV_LAYER_CONFIG: [256, 128, 64]
    DECONV_KERNEL: [4, 4, 4]
    HEAD_CONV: 64
INPUT:
  IMAGE_SIZE: 512
DATASETS:
  TRAIN: ("voc_2007_trainval", "voc_2012_trainval")
  TEST: ("voc_2007_test", )
SOLVER:
  NAME: 'ADAM_optimizer'
  MAX_ITER: 40000
  LR_STEPS: [25000, 32000]
  WARMUP_ITERS: 0
  WEIGHT_DECAY: 1e-4
  GAMMA: 0.1
  BATCH_SIZE: 32
  LR: 1.25e-4
OUTPUT_DIR: '/outputs/resnetdcn18_CNetHead_voc0712'

**NOTE: Change the output dir to your preferred path**
```
Once the config file in present, the rest of the process is the same:
```
# example (config file naming convention - given above)
python train.py --config-file configs/CenterNetHead/resnetdcn_CNet_voc0712.yaml
```

### 4. Evaluate OD as a Service
Similar to the training, the evaluation is also controlled by the same config file.
```
# Example: Backbone with SSD Head
python test.py --config-file configs/ssd/resnet_ssd512_voc0712.yaml

# Example: Backbone with CenterNet Head:
python test.py --config-file configs/CenterNetHead/resnetdcn_CNet_voc0712.yaml
```
###### Once the evaluation process completes, the output should look like this:
```
mAP: 0.7693
aeroplane       : 0.7782
bicycle         : 0.8515
bird            : 0.7722
boat            : 0.6523
bottle          : 0.5537
bus             : 0.8391
car             : 0.8614
cat             : 0.8643
chair           : 0.6173
cow             : 0.8318
diningtable     : 0.7492
dog             : 0.8510
horse           : 0.8660
motorbike       : 0.8449
person          : 0.8020
pottedplant     : 0.5305
sheep           : 0.7672
sofa            : 0.7589
train           : 0.8452
tvmonitor       : 0.7482
```

### 5. Inference with OD as a Service:
```
# Arguments:
a) --config-file : the same ones used to train and evaluate
b) --images_dir : the directory where the test images are
c) --ckpt : the directory where the trained checkpoints are present

python demo.py --config-file configs/resnet_ssd512_voc0712.yaml --images_dir <you test data path> --ckpt <folder>/<trained_checkpoint_name>.pth
```
###### The output of the process should look like:
```
(0001/0005) 004101.jpg: objects 01 | load 010ms | inference 033ms | FPS 31
(0002/0005) 003123.jpg: objects 05 | load 009ms | inference 019ms | FPS 53
(0003/0005) 000342.jpg: objects 02 | load 009ms | inference 019ms | FPS 51
(0004/0005) 008591.jpg: objects 02 | load 008ms | inference 020ms | FPS 50
(0005/0005) 000542.jpg: objects 01 | load 011ms | inference 019ms | FPS 53
```