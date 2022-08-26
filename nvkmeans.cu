//nvkmeans

#include <thrust/device_vector.h>
#include "kmeans.h"

#include <iostream>
extern "C"
{
	using std::cout;
	using std::endl;
 __attribute__((visibility ("default"))) int fit(
	//input
	int maxIter, int n, int d, int k, double *dataset, 
	//input initial center, output final center
	double *centers,
	//output
	int *labels

	)
{
	cout<<"[nvkmeans] n="<<n<<" d="<<d<<" k="<<k<<endl;	
	thrust::device_vector<double> *data_v[1];
    thrust::device_vector<int> *labels_v[1];
    thrust::device_vector<double> *centroids_v[1];
    thrust::device_vector<double> *distances_v[1];
	cout<<"[nvkmeans]creating device vectors"<<endl;
    data_v[0]      = new thrust::device_vector<double>(dataset, dataset+n*d);
    centroids_v[0] = new thrust::device_vector<double>(centers,centers+k*d);
    
	distances_v[0] = new thrust::device_vector<double>(n);
    labels_v[0]    = new thrust::device_vector<int>(n);

	bool init_from_labels = false;
	double threshold = 1e-6;
	int n_gpu=1;
	cout<<("[nvkmeans]running kmeans")<<endl;
	int iter = kmeans::kmeans(maxIter, n, d, k, data_v, labels_v, centroids_v, distances_v, n_gpu, init_from_labels,threshold);
	cout<<("[nvkmeans]copy back data")<<endl;
	int* labels_dv= thrust::raw_pointer_cast(labels_v[0]->data());
	cudaError_t ret=cudaMemcpy(labels, labels_dv, n * sizeof(int), cudaMemcpyDeviceToHost);
	if(ret!=cudaSuccess)
		cout << '[nvkmeans] copy back labels failed.' << endl;
	double *centers_dv = thrust::raw_pointer_cast(centroids_v[0]->data());
	cudaError_t ret=cudaMemcpy(centers, centers_dv, k*d * sizeof(double), cudaMemcpyDeviceToHost);
	if(ret!=cudaSuccess)
		cout << '[nvkmeans] copy back centers failed.' << endl;
	double *centers_dv = thrust::raw_pointer_cast(centroids_v[0]->data());
	cout<<"labels:";
	for(int i=0;i<=20;++i){cout<<(*labels_v[0])[i]<<',';}
	cout<<endl;
	cout<<("[nvkmeans]cleaning")<<endl;
	//clean
	delete (data_v[0]);
	delete(labels_v[0]);
    delete(centroids_v[0]);
    delete(distances_v[0]);
	cout<<("[nvkmeans]done")<<endl;
	return iter;
}

}
