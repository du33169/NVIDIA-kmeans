//nvkmeans

#include <thrust/device_vector.h>
#include "kmeans.h"

void random_labels(thrust::device_vector<int>& labels, int n, int k) {
    thrust::host_vector<int> host_labels(n);
    for(int i = 0; i < n; i++) {
        host_labels[i] = rand() % k;
    }
    labels = host_labels;
}

#include <cstdio>
extern "C"
{
 __attribute__((visibility ("default"))) int fit(
	//input
	int maxIter, int n, int d, int k, double *dataset, 
	//input initial center, output final center
	double *centers,
	//output
	int *labels

	)
{
	thrust::device_vector<double> *data_v[1];
    thrust::device_vector<int> *labels_v[1];
    thrust::device_vector<double> *centroids_v[1];
    thrust::device_vector<double> *distances_v[1];
	printf("[nvkmeans]creating device vectors\n");
    data_v[0]      = new thrust::device_vector<double>(dataset, dataset+n*d);
    centroids_v[0] = new thrust::device_vector<double>(centers,centers+k*d);
    
	distances_v[0] = new thrust::device_vector<double>(n);
    labels_v[0]    = new thrust::device_vector<int>(n);

	bool init_from_labels = false;
	double threshold = 1e-4;
	int n_gpu=1;
	printf("[nvkmeans]running kmeans\n");
	int iter = kmeans::kmeans(maxIter, n, d, k, data_v, labels_v, centroids_v, distances_v, n_gpu, init_from_labels,threshold);
	printf("[nvkmeans]copy back data\n");
	int* labels_dv= thrust::raw_pointer_cast(labels_v[0]->data());
	cudaMemcpy(labels, labels_dv, n * sizeof(int), cudaMemcpyDeviceToHost);
	double* centers_dv= thrust::raw_pointer_cast(centroids_v[0]->data());
	cudaMemcpy(centers, centers_dv, k*d * sizeof(double), cudaMemcpyDeviceToHost);
	printf("[nvkmeans]cleaning\n");
	//clean
	delete (data_v[0]);
	delete(labels_v[0]);
    delete(centroids_v[0]);
    delete(distances_v[0]);
	printf("[nvkmeans]done\n");
	return iter;
}

}