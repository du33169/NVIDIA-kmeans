//nvkmeans
#include "nvkmeans.h"

#include <thrust/device_vector.h>
#include "kmeans.h"

void random_labels(thrust::device_vector<int>& labels, int n, int k) {
    thrust::host_vector<int> host_labels(n);
    for(int i = 0; i < n; i++) {
        host_labels[i] = rand() % k;
    }
    labels = host_labels;
}

extern "C"
{
int fit(
	//input
	int maxIter, int n, int d, int k, double *dataset, 
	
	//output
	double *centers,
	int *labels

	)
{
	thrust::device_vector<double> *data_v[1];
    thrust::device_vector<int> *labels_v[1];
    thrust::device_vector<double> *centroids_v[1];
    thrust::device_vector<double> *distances_v[1];

    data_v[0]      = new thrust::device_vector<double>(dataset, dataset+n*d);
    centroids_v[0] = new thrust::device_vector<double>(k*d);
    
	distances_v[0] = new thrust::device_vector<double>(n);
    labels_v[0]    = new thrust::device_vector<int>(n);

	random_labels(*labels_v[0], n, k);
	bool init_from_labels = true;
	double threshold = 1e-4;
	int iter = kmeans::kmeans(iterations, n, d, k, data_v, labels_v, centroids_v, distances_v, n_gpu, init_from_labels,1e-4);

	int* labels_dv= thrust::raw_pointer_cast(labels_v[0]->data());
	cudaMemcpy(labels, labels_dv, n * sizeof(int), cudaMemcpyDeviceToHost);
	double* centers_dv= thrust::raw_pointer_cast(centroids_v[0]->data());
	cudaMemcpy(centers, centers_dv, k*d * sizeof(int), cudaMemcpyDeviceToHost);
	//clean
	delete (data[0]);
	delete(labels[0]);
    delete(centroids[0]);
    delete(distances[0]);
	return iter;
}

}