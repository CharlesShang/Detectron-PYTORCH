# Detectron-PYTORCH
Play with state-of-the-art pedestrian detectors.

## Requirements
1. [pytorch](http://pytorch.org/)
```
pip install http://download.pytorch.org/whl/cu80/torch-0.1.12.post2-cp27-none-linux_x86_64.whl
pip install torchvision
```
2. [tensorflow](https://www.tensorflow.org/install/install_linux#InstallingAnaconda) (only `tensorboard` is used)
```
pip install --ignore-installed --upgrade \
 https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.1.0-cp27-none-linux_x86_64.whl
```
3. [visdom](https://github.com/facebookresearch/visdom/)
```
pip install visdom
```
## Installation
1. Compiling libs
```bash
cd ./libs
make
```
2. citypersons and cityscape dataset: download coco and extract zips under `./data/citypersons/`, so your dirs look like,
```bash
./data/
        data/citypersons/{annotations}
        data/citypersons/cityscape
            data/citypersons/cityscape/leftImg8bit/{train|val|test}
            data/citypersons/cityscape/gtFine/{train|val|test}
```


## Create citypersons extended dataset
```bash
    python libs/datasets/citypersons2.py
```
## Training and evaluation models
```bash
sh ./exper/citypersons/retinanet/train_test_resnet50_softmax.sh
```