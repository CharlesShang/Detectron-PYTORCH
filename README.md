# Detectron-PYTORCH
Play with state-of-the-art pedestrian detectors.

## Requirements
1. [pytorch](http://pytorch.org/)
2. [tensorflow](https://www.tensorflow.org/install/install_linux#InstallingAnaconda) (only `tensorboard` is used)
3. tensorboardX
    ```
    pip install tensorboardX
    ```
4. [visdom](https://github.com/facebookresearch/visdom/)
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