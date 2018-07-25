#!/usr/bin/env bash

train_dir=exper/voc/retinanet/train_dir_sigmoid
lr=0.001
train_log=${train_dir}/log_train_lr${lr}_resnet101_sigmoid.log
test_log=${train_dir}/log_test_lr${lr}_resnet101_sigmoid.log
restore=${train_dir}/retinanet_resnet101_pascal_voc_ep9.h5

mkdir -p ${train_dir}

python train/train.py --cfg exper/voc/retinanet/train_test_resnet101_sigmoid.yaml \
                    --train_dir ${train_dir} \
                    --focal_loss \
                    --max_epoch 10 \
                    --lr ${lr} \
                    2>&1 | tee ${train_log}

python test/test.py --cfg exper/voc/retinanet/train_test_resnet101_sigmoid.yaml \
    --train_dir ${train_dir} \
    --restore ${restore} \
    2>&1 | tee ${test_log}