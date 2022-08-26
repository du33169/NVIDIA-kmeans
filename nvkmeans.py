from ctypes import *
import numpy as np
import numpy.random
import torch
from numpy.ctypeslib import as_ctypes
from sklearn.cluster import kmeans_plusplus
nvkmeans = CDLL("libnvkmeans.so")
class NVKmeans:

	def __init__(self,k,init) -> None:
		self.k=k
		self.init=init

	def init_centers(self,X)->np.ndarray:
		X=X.numpy()
		if self.init=='k-means++':
			centers, indices = kmeans_plusplus(X, n_clusters=self.k, random_state=0)
		elif self.init=='random':
			ind=np.random.randint(X.shape[0],size=self.k)
			centers=X[ind]
		else:
			assert False, f"[NVKmeans] unhandled init type, expected 'k-means++' or 'random', got {self.init} "
		return centers

	def fit_predict(self,X:torch.Tensor,maxIter)-> torch.Tensor: # return labels
		print(X.shape)
		n,d= X.shape
		# using kmeans++ to select initial centers
		print('[py]kmeans++...')
		centers=self.init_centers(X)
		centers=centers.ravel()
		centers=centers.astype(np.double)
		labels=np.empty(n,dtype=np.int32)

		print('[py]invoking clib...')
		self.iter=nvkmeans.fit(
			c_int(maxIter),c_int(n),c_int(d),c_int(self.k),
			as_ctypes(X.numpy().ravel().astype(np.double)),
			as_ctypes(centers),
			as_ctypes(labels),
			)
		print('labels:',np.unique(labels))
		centers.reshape((self.k,d))
		self.centers=torch.from_numpy(centers)
		print('[py]done')
		return torch.from_numpy(labels)

