all: anchor_target_kernel.cu anchor_target_kernel.h
	nvcc -c -o anchor_target.cu.o anchor_target_kernel.cu -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -arch=sm_52

clean:
	rm anchor_target.cu.o
