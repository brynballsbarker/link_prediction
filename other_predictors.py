import numpy as np
import networkx as nx
from base_predictor import basePredictor


class shortestPath(basePredictor):
    """
    predicts links with shortest path
    """
    def __init__(self, G):
        basePredictor.__init__(self, G)
        
    def compute_scores(self):
        lengths = nx.shortest_path_length(self.G)
        ns = [n for n in lengths]        
        #print(ns)
        scores = np.zeros((self.num_nodes, self.num_nodes))
        rev_dict = self.node_names_to_inds()
        for n1 in lengths:
            for n2 in lengths[n1]:
                scores[rev_dict[n1]][rev_dict[n2]] = lengths[n1][n2]
        return scores
    
class commonNeighbors(basePredictor):
    """
    predicts links with most common neighbors
    """
    def __init__(self, G):
        basePredictor.__init__(self, G)
        
    def compute_scores(self):
        A = self.get_adj()
        return A @ A
    
"""
class hittingTime(basePredictor):

"""
    
"""
class spectralEmbed(basePredictor):

"""
