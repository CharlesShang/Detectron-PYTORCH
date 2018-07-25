#!/usr/bin/env bash

train_dir=exper/citypersons/maskrcnn/train_dir/$(date '+%d-%m-%Y')
lr=0.005
train_log=${train_dir}/log_train_lr${lr}_resnet50_softmax.log
test_log=${train_dir}/log_test_lr${lr}_resnet50_softmax.log
restore=${train_dir}/maskrcnn_resnet50_citypersons_ep12.h5

mkdir -p ${train_dir}

python train/train.py --cfg exper/citypersons/maskrcnn/train_test_resnet50_softmax.yaml \
                    --train_dir ${train_dir} \
                    --no-focal_loss \
                    --max_epoch 16 \
                    --lr ${lr} \
                    2>&1 | tee ${train_log}

python test/test.py --cfg exper/citypersons/maskrcnn/train_test_resnet50_softmax.yaml \
    --train_dir ${train_dir} \
    --restore ${train_dir}/maskrcnn_resnet50_citypersons_ep8.h5 \
    2>&1 | tee -a ${test_log}

python test/test.py --cfg exper/citypersons/maskrcnn/train_test_resnet50_softmax.yaml \
    --train_dir ${train_dir} \
    --restore ${train_dir}/maskrcnn_resnet50_citypersons_ep12.h5 \
    2>&1 | tee -a ${test_log}

python test/test.py --cfg exper/citypersons/maskrcnn/train_test_resnet50_softmax.yaml \
    --train_dir ${train_dir} \
    --restore ${train_dir}/maskrcnn_resnet50_citypersons_ep14.h5 \
    2>&1 | tee -a ${test_log}
