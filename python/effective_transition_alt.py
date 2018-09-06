from scipy.sparse import linalg as la
import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix
from base_predictor import basePredictor
import helper_functions as hf


class effectiveTransitionAlt(basePredictor):
	def __init__(self, G):
		basePredictor.__init__(self, G)
		return 
	
	def compute_scores(self):
		A = self.get_adj()
		r = list(range(self.num_nodes))
		eigval = hf.scipy_eigsh(A)
		
		scores = np.zeros((self.num_nodes, self.num_nodes))
		for i in range(1,self.num_nodes):
			for j in range(i):
				if A[i,j] < 1:
					first = hf.partition(A,np.array([i,j]),np.array([i,j]))
					comp = hf.complement(r,[i,j])
					second = hf.partition(A,np.array([i,j]),comp) 
					temp = hf.partition(A, comp, comp)
					third = np.linalg.inv(hf.partition(A, comp, comp)-eigval*np.identity(self.num_nodes-2))
					fourth = hf.partition(A, comp,np.array([i,j]))
					#print(first.shape, second.shape, third.shape, fourth.shape)
					okay = second @ third @ fourth
					this = first-okay
					scores[i,j] = this[0,1]		
		return scores
		
class weightedEffectiveTransitionAlt(basePredictor):
	def __init__(self, G):
		basePredictor.__init__(self, G)
		return 
	
	def compute_scores(self):
		A = self.get_adj()
		degrees = np.sum(A, axis=1)
		M = A/degrees
		r = list(range(self.num_nodes))
		eigval = hf.scipy_eigsh(M)
		
		scores = np.zeros((self.num_nodes, self.num_nodes))
		for i in range(1,self.num_nodes):
			for j in range(i):
				if A[i,j] == 0:
					first = hf.partition(M,np.array([i,j]),np.array([i,j]))
					comp = hf.complement(r,[i,j])
					second = hf.partition(M,np.array([i,j]),comp) 
					temp = hf.partition(M, comp, comp)
					third = np.linalg.inv(hf.partition(M, comp, comp)-eigval*np.identity(self.num_nodes-2))
					fourth = hf.partition(M, comp,np.array([i,j]))
					#print(first.shape, second.shape, third.shape, fourth.shape)
					okay = second @ third @ fourth
					this = first-okay
					scores[i,j] = this[0,1]		
		return scores
