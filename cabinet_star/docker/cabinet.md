# Cabinet

- [Cabinet](#cabinet)
  - [19.07 - C++ Inference](#1907---c-inference)
    - [Docker](#docker)
      - [Installs](#installs)
        - [Dependencies](#dependencies)
        - [Prepare files to be copied into the docker image](#prepare-files-to-be-copied-into-the-docker-image)
        - [Pull and setup docker image](#pull-and-setup-docker-image)
        - [Save container to a new image](#save-container-to-a-new-image)
        - [Run inference](#run-inference)

## 19.07 - C++ Inference

### Docker

#### Installs

##### Prepare files to be copied into the docker image

```Bash
cd /home
git clone ssh://git@navbitbucket01.navinfo.eu:7999/one/cabinet_star.git
cd cabinet_star
git checkout -t origin/multinet_update

# Generate application binary.
cd cpp/src
mkdir build
cd build
cmake ..
make -j

# Generate CaBiNet TorchScript from the model.
cd ../../scripts
python save_torchscript.py --resume=/home/backup/models/cabinet/shabbir/cleaned_up/runs/mapillary/shelfnet/cabinet_base_mapillary_resnet101_0012/checkpoint_150.pth.tar --crop-size=1024 --base-size=1024 --data-folder=/home/backup/data/cabinet/mapillary/
```

##### Pull and setup docker image

```bash
docker pull nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04
nvidia-docker run --shm-size=20g --ulimit memlock=-1 --ulimit stack=67108864 --rm --name=nvidia -v /home/cabinet_star:/home/host -it nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

# Inside the container.
# Prepare build environment.
cd
apt update
apt install -y cmake-curses-gui wget unzip git
# libtorch
wget https://download.pytorch.org/libtorch/nightly/cu100/libtorch-shared-with-deps-latest.zip
unzip libtorch-shared-with-deps-latest.zip -d /usr/local
rm libtorch-shared-with-deps-latest.zip
# args
git clone https://github.com/Taywee/args.git
cd args
make install
rm -r args/
# opencv. Needs to provide infos for time-zone.
apt install -y libopencv-dev
dpkg -i /host/debs/opencv-server_3.4.1-5_amd64.deb
# anaconda for nlohmann_json
bash /host/debs/Anaconda3-2019.07-Linux-x86_64.sh
bash
conda install -c conda-forge nlohmann_json
#
# Copy files from host
cd
cp /host/cabinet_star/cpp/src/build/cabinet_cpp .
cp -r /host/cabinet_star/cpp/src .
cp -r /host/cabinet_star/cpp/resources/cabinet.pt .
cp -r /host/cabinet_star/cpp/resources/config.json .
rm -r src/build
```

##### Save container to a new image

- Outside of container.

```Bash
docker ps
docker commit 6f6d7395d63e nvcr.io/navinfo/aicv_deliveries/cabinet_cpp:19.07
```

##### Run inference

```Bash
nvidia-docker run --shm-size=20g --ulimit memlock=-1 --ulimit stack=67108864 --rm --name=nvidia -v /home:/host -it nvcr.io/navinfo/aicv_deliveries/cabinet_cpp:19.07
cd
./cabinet_cpp --classes_config_file=config.json --data_path=/host/backup/img/cabinet/same_img/ --model_file=cabinet.pt --output_path=/host/nie/cabinet_star/cpp/generated/cpp/outputs --generate_binaries
```
