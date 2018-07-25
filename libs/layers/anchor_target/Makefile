all: src/cuda/anchor_target.cu.o
	python build.py

src/cuda/anchor_target.cu.o: src/cuda/anchor_target_kernel.cu
	$(MAKE) -C src/cuda

clean:
	$(MAKE) -C src/cuda clean
