# External tool requirements
The projects needs exiftool as an external tool. Download the latest version from:
```sh
https://exiftool.org/Image-ExifTool-12.01.tar.gz
```
Extract the tool into a local folder and follow the steps in the README.md of the tool.

# Optional tool install
Latest version of ffmpeg (an older version is already in the environment)
```bash
sudo apt-get install ffmpeg
``` 

# How to install?


```sh
git clone ssh://git@navbitbucket01.navinfo.eu:7999/cd/crdv2.git
cd crdv2
git submodule update --depth 1 --init --recursive
conda env create -f environment.yml
./misc_build.sh
```

# How to run?

```sh
conda activate crd
```


## from GPS txt files
```sh
python main.py \    
    /data/input/datasets/crd_dataset/runA/1001-2019-09-241913/2019-09-241913_GPS.txt \
    /data/input/datasets/crd_dataset/runB/1101-2019-10-081806/2019-10-081806_GPS.txt \
    -f default
```

## from images
```sh
python main.py \    
    /data/aiim/change_detection/ChangeDetection_sampledata/Augsburg/w33/ \
    /data/aiim/change_detection/ChangeDetection_sampledata/Augsburg/w35/ \
    -f blackvue_images
```

## from video
```sh
python main.py \    
    /data/aiim/change_detection/comparison00-essen/w2719/videos \
    /data/aiim/change_detection/comparison00-essen/w3519/videos \
    -f blackvue_video
```

### separate cache dir
```sh
python main.py \    
    /data/aiim/change_detection/ChangeDetection_sampledata/Augsburg/w33/ \
    /data/aiim/change_detection/ChangeDetection_sampledata/Augsburg/w35/ \
    -f blackvue_images -c <YOUR OUTPUT DIR>
```

### with no speed correction (NOTE: more inaccurate but much faster)
```sh
python main.py \    
    /data/aiim/change_detection/ChangeDetection_sampledata/Augsburg/w33/ \
    /data/aiim/change_detection/ChangeDetection_sampledata/Augsburg/w35/ \
    -f blackvue_images -nsc
```
