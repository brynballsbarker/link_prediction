import numpy as np
import networkx as nx
import algebraic_connectivity as ac
import numpy.linalg as la
import scipy.sparse.linalg as sla
import time

def tracemin_solver(L,dim=1, method='pcg', tol=1e-8):
    """
    Find the dim smallest nonzero eigen-values and vectors of L using the tracemin algorithm
    """
    n = L.shape[0]
    q = min(2*dim, L.shape[0]-1)
    X = np.asmatrix(np.random.normal(size=(n, q)))
    #apparently searching a higher dimensional space sometimes gives better results
    normalized = False
    sig, V = ac._tracemin_fiedler(L, X, normalized, tol, method, num_vecs = dim)
    return sig[:dim], V[:,:dim]

def scipy_eigh(L, dim=1, tol=1e-8):
    """
    Find the dim smallest nonzero eigen-values and vectors of L using scipy's built-in
    """
    L = L.astype(np.float64)
    sigma, V = sla.eigsh(L, k=dim+1, which='SM', tol=tol, return_eigenvectors=True)
    return sigma[1:], V[:,1:]

def time_solver(mat, solver, dim=1, **kwargs):
    start = time.time()
    sigma, V = solver(mat, dim=dim, **kwargs)
    print("here in time solver")
    print("sigma ", sigma)

    return time.time()-start