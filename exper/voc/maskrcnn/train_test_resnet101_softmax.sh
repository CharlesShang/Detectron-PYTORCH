#!/usr/bin/env bash

train_dir=exper/voc/maskrcnn/train_dir
lr=0.001
train_log=${train_dir}/log_train_lr${lr}_resnet101_softmax.log
test_log=${train_dir}/log_test_lr${lr}_resnet101_softmax.log
restore=${train_dir}/retinanet_resnet101_pascal_voc_ep14.h5
mkdir -p ${train_dir}

python train/train.py --cfg exper/voc/maskrcnn/train_test_resnet101.yaml \
                    --train_dir ${train_dir} \
                    --focal_loss \
                    --max_epoch 15 \
                    --lr ${lr} \
                    2>&1 | tee ${train_log}

python test/test.py --cfg exper/voc/maskrcnn/train_test_resnet101.yaml \
    --train_dir ${train_dir} \
    --restore ${restore} \
    2>&1 | tee ${test_log}