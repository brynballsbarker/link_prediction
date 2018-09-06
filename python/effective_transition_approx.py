from scipy.sparse import linalg as la
import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix
from base_predictor import basePredictor
import helper_functions as hf


class effectiveTransitionApprox(basePredictor):
	def __init__(self, G, cut=10):
		self.cut=cut
		basePredictor.__init__(self, G)
		return 
	
	def compute_scores(self):
		A = self.get_adj()
		r = list(range(self.num_nodes))
		eigval = hf.scipy_eigsh(A)
		
		scores = np.zeros((self.num_nodes, self.num_nodes))
		for i in range(1,self.num_nodes):
			for j in range(i):
				if A[i,j] == 0:
					comp = hf.complement(r,[i,j])
					a = hf.partition(A, comp,np.array([i,j]))
					b = hf.partition(A, np.array([i,j]), comp)
					q = hf.partition(A, comp, comp)
					to_sum = [np.eye(q.shape[0])]
					for i in range(self.cut):
						to_sum.append(to_sum[i]@q)
					Nk = sum(to_sum)
					if i==1 and j==0:
						print(b.shape, Nk.shape, a.shape)
					scores[i,j] = (b@Nk@a)[0,1]				
				
					"""
					first = hf.partition(A,np.array([i,j]),np.array([i,j]))
					comp = hf.complement(r,[i,j])
					second = hf.partition(A,np.array([i,j]),comp) 
					temp = hf.partition(A, comp, comp)
					third = np.linalg.inv(hf.partition(A, comp, comp)-eigval*np.identity(self.num_nodes-2))
					fourth = hf.partition(A, comp,np.array([i,j]))
					okay = second @ third @ fourth
					this = first-okay
					scores[i,j] = this[0,1]	
					"""	
		return scores
		
class weightedEffectiveTransitionApprox(basePredictor):
	def __init__(self, G, cut=10, directed=False):
		basePredictor.__init__(self, G, directed)
		self.cut = cut
		return 
	
	def compute_scores(self):
		A = self.get_adj()
		degrees = np.sum(A, axis=1)
		M = A/degrees
		r = list(range(self.num_nodes))
		eigval = hf.scipy_eigsh(M)
		
		scores = np.zeros((self.num_nodes, self.num_nodes))
		if self.directed:
			for i in range(self.num_nodes):
				for j in range(i):
					if A[i,j] == 0:#i != j:
					
						comp = hf.complement(r,[i,j])
						a = hf.partition(M, comp,np.array([i,j]))
						b = hf.partition(M, np.array([i,j]), comp)
						q = hf.partition(M, comp, comp)
						to_sum = [np.eye(q.shape[0])]
						for i in range(self.cut):
							to_sum.append(to_sum[i]@q)
						Nk = sum(to_sum)
						scores[i,j] = (b@Nk@a)[0,1]		
		else:
			for i in range(1,self.num_nodes):
				for j in range(i):
					if A[i,j] == 0:#i != j:
					
						comp = hf.complement(r,[i,j])
						a = hf.partition(M, comp,np.array([i,j]))
						b = hf.partition(M, np.array([i,j]), comp)
						q = hf.partition(M, comp, comp)
						to_sum = [np.eye(q.shape[0])]
						for i in range(self.cut):
							to_sum.append(to_sum[i]@q)
						Nk = sum(to_sum)
						scores[i,j] = (b@Nk@a)[0,1]	
					
						"""
						first = hf.partition(M,np.array([i,j]),np.array([i,j]))
						comp = hf.complement(r,[i,j])
						second = hf.partition(M,np.array([i,j]),comp) 
						temp = hf.partition(M, comp, comp)
						third = np.linalg.inv(hf.partition(M, comp, comp)-eigval*np.identity(self.num_nodes-2))
						fourth = hf.partition(M, comp,np.array([i,j]))
						okay = second @ third @ fourth
						this = first-okay
						scores[i,j] = this[0,1]		
						"""
		return scores
