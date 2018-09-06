import numpy as np
import networkx as nx

class basePredictor():
    def __init__(self, G, directed=False):
        """
        basic link predictor
        """
        self.G = G
        self.node_list = G.nodes()
        self.num_nodes = len(self.node_list)
        self.num_edges = G.number_of_edges()
        self.directed = directed
        self.A = None
        self.L = None
        self.scores = None
        self.links = None
        self.ncorrect = None
        self.acc = None
        return
    
    def empty_like(self):
        return np.zeros((self.num_nodes, self.num_nodes))
    
    def get_adj(self):
        self.A = np.array(nx.to_numpy_matrix(self.G))
        return self.A
    
    def get_laplacian(self):
        A = self.get_adj()
        self.L = np.diag(np.sum(A, axis=0)) - A
        return self.L
    
    def node_names_to_inds(self):
        return {self.node_list[i]: i for i in range(self.num_nodes)}
    
    def validate(self, true_graph, min_links=None, force_exact=False, return_links=False, **kwargs):
        """
        predicts new links and validates against true_graph
        if min_links=None: predicts only best scores
        if force_exact=True: predict exactly min links
        if return_links=True: return links instead of number
        """
        links = self.predict(min_links=min_links, **kwargs)
        
        if force_exact:
            if min_links is not None:
                num_to_predict = min_links
                while len(links) < min_links:
                    num_to_predict *= 2
                    links = self.predict(min_links=num_to_predict, **kwargs)
                    
            ncorrect = 0
            links = links[:min_links]
            for link in links:
                if true_graph.has_edge(link[0], link[1]):
                    ncorrect += 1
                    
        else:
            ncorrect = 0
            for link in links:
                if true_graph.has_edge(link[0], link[1]):
                    ncorrect += 1
        
        self.links = links
        self.ncorrect = ncorrect
        if return_links:
            return links, ncorrect
        else:
            return len(links), ncorrect
        
    def compute_scores(self):
        self.scores = np.ones((self.num_nodes, self.num_nodes))
        return self.scores
        
    def predict(self, min_links=None, score_tolerance=1e-10, score_inf=1e18):
        """
        predicts new links in a graph
        """
        scores = self.compute_scores()
        A = self.get_adj()
        
        if not self.directed:
        	scores[np.triu_indices(scores.shape[0])] = score_inf
        scores[(A>0)] = score_inf
        np.fill_diagonal(scores, score_inf)
        min_score = np.min(scores)
        flat_scores = scores.flatten()
        dest = np.zeros(flat_scores.size)
        dest[:] = flat_scores[:]
        if min_links is not None:
        	sorted_inds = np.argsort(flat_scores)[:min_links]
        	last_score = flat_scores[sorted_inds[-1]]
        	other_inds = np.argwhere(np.abs(dest-last_score) < score_tolerance)
        	all_inds = np.concatenate((sorted_inds[:min_links], other_inds.flatten()))
        	to, frm = np.unravel_index(all_inds, dims=A.shape)
        else:
        	all_inds = np.argwhere(np.abs(dest-min_score) < score_tolerance)
        	to, frm = np.unravel_index(all_inds.flatten(), dims=A.shape)

            
        return set([(self.node_list[i], self.node_list[j]) for i,j in zip(to,frm)])
    
    def accuracy(self, true_graph=None, k=None, **kwargs):
        if self.ncorrect == None:
            nlinks, ncorrect = self.validate(true_graph, min_links=k, force_exact=False,**kwargs)
        else:
            nlinks = len(self.links)
            ncorrect = self.ncorrect
        
        self.acc = ncorrect/nlinks
        return self.acc