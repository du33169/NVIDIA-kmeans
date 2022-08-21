CUDA_ARCH ?= compute_80 
# for detail, execute command below:
# nvcc --help | grep -F 'gpu-architecture <arch>' -A 27

#CUB_HOME ?= ./cub   # skip for higher version of CUDA

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

clean:
	rm -f *.o test
