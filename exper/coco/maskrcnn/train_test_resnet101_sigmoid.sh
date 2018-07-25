#!/usr/bin/env bash

train_dir=exper/coco/maskrcnn/train_dir_sigmoid
lr=0.001
train_log=${train_dir}/log_train_lr${lr}_resnet101_sigmoid.log
test_log=${train_dir}/log_test_lr${lr}_resnet101_sigmoid.log
restore=${train_dir}/maskrcnn_resnet101_coco_ep14.h5

mkdir -p ${train_dir}

python train/train.py --cfg exper/coco/maskrcnn/train_test_resnet101.yaml \
                    --train_dir ${train_dir} \
                    --focal_loss \
                    --max_epoch 15 \
                    --lr ${lr} \
                    --activation sigmoid \
                    2>&1 | tee ${train_log}

python test/test.py --cfg exper/coco/maskrcnn/train_test_resnet101.yaml \
    --train_dir ${train_dir} \
    --restore ${restore} \
    --activation sigmoid \
    2>&1 | tee ${test_log}