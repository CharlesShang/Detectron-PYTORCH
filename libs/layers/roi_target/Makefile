all: src/cuda/roi_target.cu.o
	python build.py

src/cuda/roi_target.cu.o: src/cuda/roi_target_kernel.cu
	$(MAKE) -C src/cuda

clean:
	$(MAKE) -C src/cuda clean
