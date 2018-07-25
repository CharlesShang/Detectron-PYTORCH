all: roi_target_kernel.cu roi_target_kernel.h
	nvcc -c -o roi_target.cu.o roi_target_kernel.cu -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -arch=sm_52

clean:
	rm roi_target.cu.o
