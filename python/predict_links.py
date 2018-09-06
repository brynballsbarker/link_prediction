import numpy as np
import networkx as nx

def delete_duplicates(pairs):
        """
        deletes duplicate pairs from list pairs
        """
        n = len(pairs)
        copy = []
        for pair in pairs:
            copy.append(sorted(pair))

        for i in range(1,n):
            if copy.count(copy[n-i]) > 1:
                del copy[n-i]
        return copy

def get_loc(pos, n):
        """
        takes position in flattened array and gives location in non-flattened
        """
        row = int(pos/n)
        col = pos % n
        return (row, col)

def backward_dict(pairs, nodes):
	n = len(pairs)
	actual = [(nodes[pair[0]],nodes[pair[1]]) for pair in pairs]
	return actual

def predict(scores, nodes, A, k):
	n = len(nodes)
	print('here')
	pred = np.asarray(np.argsort(-1*(scores-10*A).reshape(n*n)))[0]
	print(pred)
	prediction = []
	for p in pred:
		prediction.append(get_loc(p,n))
	prediction = delete_duplicates(prediction)[:k]
	return backward_dict(prediction, nodes)

def accuracy(prediction, to_predict):
	return len(set(prediction)&set(to_predict))/len(to_predict)
