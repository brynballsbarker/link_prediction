import networkx as nx
import numpy as np
import scipy.linalg as la
from base_predictor import basePredictor
import helper_functions as hf
from closest_points import closest_points
import eigen_solvers as es

class spectralPred(basePredictor):
	def __init__(self, G):
		basePredictor.__init__(self,G)
		self.numEigenSpaces = 1
		return
		
	def set_num_spaces(self, spaces):
		self.numEigenSpaces = spaces
		return
		
	def compute_scores(self):
		L = self.get_laplacian()
		node_locs = hf.smallest_laplacian_eigenspaces(L, num_spaces=self.numEigenSpaces)
		n = self.num_nodes
		scores = np.zeros((n,n))
		
		ndims = node_locs.shape[1]
		for i in range(ndims):
			scores += (node_locs[:,i].reshape((n,1)) - node_locs[:,i].reshape((1,n))) **2
		return np.sqrt(scores)
		
class spectralEmbed(basePredictor):
	def __init__(self, G, dim=1, esolver='tracemin', directed=False,**kwargs):
		basePredictor.__init__(self,G,directed)
		self.dim = dim
		self._compute_embedding(esolver, **kwargs)
		return
		
	def _compute_embedding(self, esolver='tracemin', **kwargs):
		solver = None
		if esolver == 'tracemin':
			solver = es.tracemin_solver
		else:
			solver = es.scipy_eigh
		sig, V = solver(nx.laplacian_matrix(self.G), dim=self.dim, **kwargs)
		self.points = V/np.sqrt(sig)
		
	def euclidean(self, min_links, dim=None):
		pairs, dists = closest_points(self.points[:,:dim], k=min_links)
		res = []
		for i in range(min_links):
			u, v = self.node_list[pairs[i][0]],self.node_list[pairs[i][1]]
			if not self.G.has_edge(u,v):
				res.append((u,v))
		return res
	
	def cosine(self, min_links):
		norms = np.sqrt(np.sum(self.points**2, axis=1))
		#Need to reshape the norms array so broadcasting will work        
		pairs, dists = closest_points(self.points / norms[:,None], k= min_links)
		res = []
		for i in range(min_links):
			u, v = self.node_list[pairs[i][0]],self.node_list[pairs[i][1]]
			if not self.G.has_edge(u,v):
				res.append((u,v))
		return res
		
	def parse_dim(self, dim):
		if dim is None or dim <= 0 or dim > self.dim: return self.dim
		
	def predict(self, min_links=None, score_tolerance = 1e-10, method="euclidean", dim=None):
		dim = self.parse_dim(dim)
		if min_links is None:
			min_links = self.num_nodes
		if method == "euclidean":
			return self.euclidean(min_links, dim=dim)
		elif method == "cosine":
			return self.cosine(min_links)
			
	# this isn't working and i don't know why
	#def validate(self, true_graph, min_links=None, **kwargs):
	#	print('here is an issue i think')
	#	return basePredictor.validate(true_graph, min_links, force_exact=True, **kwargs)
 