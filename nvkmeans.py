from ctypes import *
import numpy as np
import torch
from numpy.ctypeslib import as_ctypes
from sklearn.cluster import kmeans_plusplus
nvkmeans = CDLL("libnvkmeans.so")
class NVKmeans:

	def __init__(self,k) -> None:
		self.k=k

	def fit_predict(self,X:torch.Tensor,maxIter)-> torch.Tensor: # return labels
		n,d= X.shape
		# using kmeans++ to select initial centers
		print('[py]kmeans++...')
		centers, indices = kmeans_plusplus(X.numpy(), n_clusters=self.k, random_state=0)
		centers=centers.ravel()
		centers=centers.astype(np.double)
		labels=np.empty((n,1),dtype=np.int32)

		print('[py]invoking clib...')
		self.iter=nvkmeans.fit(
			c_int(maxIter),c_int(n),c_int(d),c_int(self.k),
			as_ctypes(X.numpy().ravel()),
			as_ctypes(centers),
			as_ctypes(labels),
			)
		centers.reshape((self.k,d))
		self.centers=torch.from_numpy(centers)
		print('[py]done')
		return torch.from_numpy(labels.ravel())

