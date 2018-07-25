#!/usr/bin/env bash

train_dir=exper/coco/maskrcnn/train_dir
lr=0.001
train_log=${train_dir}/log_train_lr${lr}_resnet101.log
test_log=${train_dir}/log_test_lr${lr}_resnet101.log
restore=${train_dir}/maskrcnn_resnet101_coco_ep14.h5

mkdir -p ${train_dir}

python train/train.py --cfg exper/coco/maskrcnn/train_test_resnet101.yaml \
                    --train_dir ${train_dir} \
                    --focal_loss \
                    --max_epoch 15 \
                    --lr ${lr} \
                    2>&1 | tee ${train_log}

python test/test.py --cfg exper/coco/maskrcnn/train_test_resnet101.yaml \
    --train_dir ${train_dir} \
    --restore ${restore} \
    2>&1 | tee ${test_log}


#python -m cProfile -o profile.out \
#     train/train.py --cfg exper/coco/maskrcnn/train_test_resnet101.yaml \
#                    --train_dir ${train_dir} \
#                    --focal_loss \
#                    --max_epoch 1 \
#                    --lr ${lr} \
#                    2>&1 | tee ${train_log}
#
## generate an image
#if [ ! -f gprof2dot.py ]; then
#	echo "Downloading ... "
#	wget https://raw.githubusercontent.com/jrfonseca/gprof2dot/master/gprof2dot.py -O gprof2dot.py
#fi
#python gprof2dot.py -f pstats profile.out | dot -Tpng -o profile3.png

