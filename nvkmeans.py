from ctypes import *
import numpy as torch
import torch
from numpy.ctypeslib import as_ctypes
nvkmeans = CDLL("libnvkmeans.so")
class NVKmeans:

	def __init__(self,k) -> None:
		self.k=k

	def fit(self,X:torch.Tensor,maxIter)-> int: # return the number of actual iterations
		n,d= (X.shape[0]), (X.shape[1])
		centers=torch.empty((n*d,1),dtype=torch.int32)
		labels=torch.empty((n,1),dtype=torch.int32)
		#ramdom generate center
		iter=nvkmeans.fit(
			c_int(maxIter),c_int(n),c_int(d),c_int(self.k),
			as_ctypes(X.ravel()),
			as_ctypes(centers),
			as_ctypes(labels),
			)
		centers.reshape((n,d))
		self.centers=centers
		return iter

	def predict(self,X:torch.Tensor):
		eudis=lambda x,y:torch.linalg.norm(x-y)
		def nearest(x)->int:
			'''return nearest label'''
			dises=torch.tensor([eudis(x,c) for c in self.centers])
			return torch.argmin(dises)
		return torch.tensor([nearest(x) for x in X])

