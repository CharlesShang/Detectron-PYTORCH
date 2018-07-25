#!/usr/bin/env bash

train_dir=exper/coco/retinanet/train_dir
lr=0.002
train_log=${train_dir}/log_train_lr${lr}_sigmoid.log
test_log=${train_dir}/log_test_lr${lr}_sigmoid.log

mkdir -p ${train_dir}

python train/train.py --cfg exper/coco/retinanet/train_resnet50_sigmoid.yaml \
                    --train_dir ${train_dir} \
                    --focal_loss \
                    --max_epoch 15 \
                    --lr ${lr} \
                    2>&1 | tee ${train_log}

python test/test.py --cfg exper/coco/retinanet/test_resnet50_sigmoid.yaml \
    --train_dir ${train_dir} \
    --restore ${restore} \
    2>&1 | tee ${train_log}