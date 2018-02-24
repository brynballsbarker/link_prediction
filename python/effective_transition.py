
from scipy.sparse import linalg as la
import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt

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

def complement(first, second):
    """
    returns the compliment of the first list in the second
    """
    second = set(second)
    return [item for item in first if item not in second]

def get_loc(pos, n):
    """
    takes position in flattened array and gives location in non-flattened
    """
    row = int(pos/n)
    col = pos % n
    return (row, col)

def partition(M, r, c):
    """
    returns the r rows and c columns of matrix M
    """
    part = []
    for x in r:
        for y in c:
            part.append(M[x,y])
    return np.array(part).reshape((len(r),len(c)))

def scipy_eigsh(M, dim=1, tol=1e-8):
    """
    returns the eigenvalue of largest magnitude corresponding to matrix M
    """
    M = M.astype(np.float64)
    sigma = la.eigsh(M, k=dim, which='LM', tol=tol, return_eigenvectors=False)
    return sigma[0]

def brute_effective_transition(M,k):
    """
    predicts k links for network with adjacency matrix M
    """
    n = M.shape[0]
    r = list(range(n))
    eigval = scipy_eigsh(M)

    R = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if i != j:
                first = partition(M,[i,j],[i,j])
                comp = complement(r,[i,j])
                second = partition(M,[i,j],comp) 
                temp = partition(M, comp, comp)
                third = np.linalg.inv(partition(M, comp, comp)-eigval*np.identity(n-2))
                fourth = partition(M, comp,[i,j])
                R[i,j] = (first - np.dot(second,np.dot(third,fourth)))[0,1]

    pred = np.asarray(np.argsort(-1*(R - 10*M).reshape(n*n)))[0]

    prediction = []
    for p in pred:
        prediction.append(get_loc(p,n))
        
    return delete_duplicates(prediction)[:k]
