from scipy.sparse import linalg as la
import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix
from base_predictor import basePredictor
import helper_functions as hf


class weightedEffectiveTransitionApprox(basePredictor):
	def __init__(self, G, steps=10):
		basePredictor.__init__(self, G)
		self.steps = steps
		return 

	def _compute(self, i, j, step, curr, path):
		if step < self.steps:
			for n in range(self.num_nodes):
				if self.M[n,j] != 0 and n != i:
					if n not in path:
						self.R[i,n] += curr*self.M[n,j]
					self.compute(i,n,step+1,self.M[n,j]*curr,path+[n])

	
	def compute_scores(self):
		A = self.get_adj()
		degrees = np.sum(A, axis=1)
		self.M = (A/degrees).T
		self.R = np.zeros((self.num_nodes, self.num_nodes))
		for i in range(self.num_nodes):
			self.compute(i, i,0, 1, [i])
		return self.R
