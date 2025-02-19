my modifications to make it work:

- Replaced all `device_malloc_allocator` to `device_allocator` in labels.h, in accordance with test.cu
- Wrote and verified a Makefile, which was not provided in the original repository

---

kmeans
======

A simple kmeans clustering implementation for double precision data,
written for CUDA GPUs.

There are two ideas here:

  1. The relabel step of kmeans relies on computing distances between
all n points (x) and all k centroids (y). This code refactors the distance
computation using the identity ||x-y||^2 = x.x + y.y - 2x.y; this
refactorization moves the x.x computation outside the kmeans loop, and
uses GEMM to compute the x.y, getting us peak performance. 
  2. The computation of new centroids can be tricky because the labels
change every iteration.  This code shows how to sort to group all points with
the same label, transforming the centroid accumulation into 
simple additions, minimizing atomic memory operations.  For many
practical problem sizes, sorting reduces the centroid computation to less
than 20% of the overall runtime of the algorithm.

The CUDA code here is purposefully non-optimized - this code is not
meant to be the fastest possible kmeans implementation, but rather to
show how using libraries like thrust and BLAS can provide reasonable
performance with high programmer productivity.

Multi-GPU version
=================
This version has been updated to use multiple GPUs attached to the same machine.
You do not need to specify the number of GPUs, the program will detect and use
them.

Prerequisites
=============
* ~~CUDA toolkit 4.2~~
* ~~CUB 1.0.2 https://github.com/NVLabs/cub~~
* tested on CUDA 11.0, but should be compatible with ealier version.

Build
=====
~~To build, edit Makefile to specify CUB_HOME, the location of your CUB files~~
~~Then call make.~~

1. Check the compute capability of your GPU [here](https://developer.nvidia.com/cuda-gpus#compute)

2. Execute the following command, and decide your `CUDA_ARCH` accordingly.
    ```bash
    nvcc --help | grep -F 'gpu-architecture <arch>' -A 27
    ```

3. Edit `CUDA_ARCH` in Makefile

4. If you need a custom version of CUB, edit  `CUB_HOME` in Makefile

5. Call make, and a executable file `test` will be created.
    ```bash
    make
    ```

Run
===
A simple test case is run when you invoke the executable 'test'.

For demonstration, test will generate and solve 3 test cases of different
sizes. At the prompt, specify 't' for a tiny test case, 'm' for a slightly 
bigger test case, and 'h' for a huge test case: 1 million points, with 50 
dimensions and 100 clusters, for 50 iterations.
