import networkx as nx
import numpy as np
import numpy.linalg as la

def shortest_path(g, A):
	"""
	predicts links using shortest path method
	"""
	print('here')
	lengths = nx.shortest_path_length(g)
	nodes = g.nodes()
	num_nodes = len(nodes)

	scores = np.zeros((num_nodes, num_nodes))
	rev_dict = {nodes[i]: i for i in range(num_nodes)}
	for n1 in lengths:
		for n2 in lengths[n1]:
			scores[rev_dict[n1]][rev_dict[n2]] = lengths[n1][n2]
	return -1*scores

def common_neighbors(g, A):
	scores = np.dot(A,A)
	return scores

def preferential_attachment(g, A):
	degrees = np.sum(A, axis=1)
	scores = np.outer(degrees, degrees)
	return scores

def jacard(g, A):
	num = np.dot(A,A)
	# finish
	return

def katz(g, A):
	return


