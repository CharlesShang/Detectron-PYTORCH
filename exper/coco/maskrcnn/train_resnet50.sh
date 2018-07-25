#!/usr/bin/env bash

train_dir=exper/coco/maskrcnn/train_dir
lr=0.001
train_log=${train_dir}/log_train_lr${lr}_resnet50.log
test_log=${train_dir}/log_test_lr${lr}_resnet50.log

mkdir -p ${train_dir}

python train/train.py --cfg exper/coco/maskrcnn/train_resnet50.yaml \
                    --train_dir ${train_dir} \
                    --focal_loss \
                    --max_epoch 15 \
                    --lr ${lr} \
                    2>&1 | tee ${train_log}

python test/test.py --cfg exper/coco/maskrcnn/test_resnet50.yaml \
    --train_dir ${train_dir} \
    --restore ${restore} \
    2>&1 | tee ${train_log}