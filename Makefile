# [CUDA_ARCH]
# specify gpu architecture, corresponding to your GPU Compute Capability
# https://developer.nvidia.com/cuda-gpus#compute
# execute: 
# $	nvcc --help | grep -F 'gpu-architecture <arch>' -A 27
# for details,

CUDA_ARCH ?= compute_80 

# [CUB_HOME]
# customize cub path, uncomment and edit if needed
# CUB_HOME ?= ./cub   

# [COMPILER]
CC   = nvcc 
ARGS = -arch=$(CUDA_ARCH) -Xptxas -v  -lineinfo

test: timer.o labels.o centroids.o kmeans.o test.cu  
	$(CC) $(ARGS) -lcublas $^ -o $@

timer.o: timer.cu timer.h
	$(CC) $(ARGS) -c timer.cu -o $@

labels.o: labels.cu labels.h
	$(CC) $(ARGS) -c labels.cu -o $@

centroids.o: centroids.cu centroids.h
	$(CC) $(ARGS) -c centroids.cu -o $@

kmeans.o: kmeans.cu kmeans.h util.h
	$(CC) $(ARGS) -c kmeans.cu -o $@

shared:
	$(CC) -arch=$(CUDA_ARCH) -v -shared -Xcompiler "-fPIC" -c kmeans.cu -o libnvkmeans.so
clean:
	rm -f *.o test
