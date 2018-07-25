#!/usr/bin/env bash

CUDA_PATH=/usr/local/cuda/
# build pycocotools
#cd datasets/pycocotools
#make
#cd -
#
#cd layers/sample/src
#ls
#echo "Compiling sample layer kernels by nvcc..."
#nvcc -c -o sample_cuda_kernel.cu.o sample_cuda_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_52
#cd ../
#python build.py
#cd ../..

cd layers/nms/src
ls
echo "Compiling nms layer by nvcc..."
nvcc -c -o nms_cuda_kernel.cu.o nms_cuda_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_52
cd ../
python build.py
cd ../../

pwd
cd layers/roi_align/src/cuda
ls
echo "Compiling roi align layer by nvcc..."
nvcc -c -o roi_align.cu.o roi_align_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_52
cd ../../
python build.py
cd ../../
pwd


cd layers/roi_target/src/cuda
ls
echo "Compiling roi_target layer by nvcc..."
nvcc -c -o roi_target.cu.o roi_target_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_52
cd ../../
python build.py
cd ../../
pwd

cd layers/anchor_target/src/cuda/
ls
echo "Compiling roi_target layer by nvcc..."
nvcc -c -o anchor_target.cu.o anchor_target_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_52
cd ../../
python build.py
cd ../../
pwd

cd layers/roi_align_tf/src/cuda
ls
echo "Compiling roi_target layer by nvcc..."
nvcc -c -o crop_and_resize_kernel.cu.o crop_and_resize_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_52
cd ../../
python build.py
cd ../../
pwd

cd datasets/kiiti_eval
pwd
echo "Compiling kitti_eval."
g++ -O3 -DNDEBUG -Wno-cpp -Wno-unused-function -Wno-unused-result -o evaluate_object.bin evaluate_object.cpp
cd -
pwd

cd datasets/pycocotools
pwd
echo "Compiling pycocotools."
make
cd -
pwd
