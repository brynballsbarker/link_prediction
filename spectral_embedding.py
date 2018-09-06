import numpy as np
import networkx as nx
#import algebraicconnectivity as ac
#import graph_utils as gu
import numpy.linalg as la
import scipy.sparse.linalg as sla
import time

from closest_points import closest_points

from functools import partial
import networkx as nx
from networkx.utils import not_implemented_for
from networkx.utils import reverse_cuthill_mckee_ordering
from re import compile


from numpy import (array, asmatrix, asarray, dot, matrix, ndarray, ones,
                       reshape, sqrt, zeros)
from numpy.linalg import norm, qr
from numpy.random import normal
from scipy.linalg import eigh, inv
from scipy.sparse import csc_matrix, spdiags
from scipy.sparse.linalg import eigsh, lobpcg

class _PCGSolver(object):
    """Preconditioned conjugate gradient method.
    """

    def __init__(self, A, M):
        self._A = A
        self._M = M or (lambda x: x.copy())

    def solve(self, B, tol):
        B = asarray(B)
        X = ndarray(B.shape, order='F')
        for j in range(B.shape[1]):
            X[:, j] = self._solve(B[:, j], tol)
        return X

    def _solve(self, b, tol):
        A = self._A
        M = self._M
        tol *= dasum(b)
        # Initialize.
        x = zeros(b.shape)
        r = b.copy()
        z = M(r)
        rz = ddot(r, z)
        p = z.copy()
        # Iterate.
        while True:
            Ap = A(p)
            alpha = rz / ddot(p, Ap)
            x = daxpy(p, x, a=alpha)
            r = daxpy(Ap, r, a=-alpha)
            if dasum(r) < tol:
                return x
            z = M(r)
            beta = ddot(r, z)
            beta, rz = beta / rz, beta
            p = daxpy(p, z, a=beta)

def _tracemin_fiedler(L, X, normalized, tol, method, num_vecs=None):
    """Compute the Fiedler vector of L using the TraceMIN-Fiedler algorithm.
    """
    n = X.shape[0]
    if num_vecs is None: num_vecs = 1

    if normalized:
        # Form the normalized Laplacian matrix and determine the eigenvector of
        # its nullspace.
        e = sqrt(L.diagonal())
        D = spdiags(1. / e, [0], n, n, format='csr')
        L = D * L * D
        e *= 1. / norm(e, 2)

    if not normalized:
        def project(X):
            """Make X orthogonal to the nullspace of L.
            """
            X = asarray(X)
            for j in range(X.shape[1]):
                X[:, j] -= X[:, j].sum() / n
    else:
        def project(X):
            """Make X orthogonal to the nullspace of L.
            """
            X = asarray(X)
            for j in range(X.shape[1]):
                X[:, j] -= dot(X[:, j], e) * e


    if method is None:
        method = 'pcg'
    if method == 'pcg':
        # See comments below for the semantics of P and D.
        def P(x):
            x -= asarray(x* X*X.T)[0, :]
            if not normalized:
                x -= x.sum() / n
            else:
                x = daxpy(e, x, a=-ddot(x, e))
            return x
        solver = _PCGSolver(lambda x: P(L * P(x)), lambda x: D * x)
    """elif method == 'chol' or method == 'lu':
        # Convert A to CSC to suppress SparseEfficiencyWarning.
        A = csc_matrix(L, dtype=float, copy=True)
        # Force A to be nonsingular. Since A is the Laplacian matrix of a
        # connected graph, its rank deficiency is one, and thus one diagonal
        # element needs to modified. Changing to infinity forces a zero in the
        # corresponding element in the solution.
        i = (A.indptr[1:] - A.indptr[:-1]).argmax()
        A[i, i] = float('inf')
        solver = (_CholeskySolver if method == 'chol' else _LUSolver)(A)
    else:
        raise nx.NetworkXError('unknown linear system solver.')"""

    # Initialize.
    Lnorm = abs(L).sum(axis=1).flatten().max()
    project(X)
    W = asmatrix(ndarray(X.shape, order='F'))
    #The converged vectors
    X_conv = asmatrix(ndarray(X.shape, order='F'))
    sig_conv = zeros(num_vecs)
    nconv = 0
    while True:
        # Orthonormalize X.
        X = qr(X)[0]
        # Compute interation matrix H.
        W[:, :] = L * X
        H = X.T * W
        sigma, Y = eigh(H, overwrite_a=True)
        # Compute the Ritz vectors.
        X = X * Y
        # Test for convergence exploiting the fact that L * X == W * Y.
        
        #Test convergence, 
        #This is really the number of consecutive vectors that converged
        max_conv = 0
        num_remaining = num_vecs-nconv
        errs = []
        for i in range(num_remaining):
            err = dasum(W * asmatrix(Y)[:, i] - sigma[i] * X[:, i]) / Lnorm
            errs.append(err)
            if err < tol:
                max_conv = i+1
            else:
                break                
        

        print(nconv, errs)
        if max_conv == num_remaining:            
            X_conv[:,nconv:nconv+max_conv] = X[:,:max_conv]
            sig_conv[nconv:nconv+max_conv] = sigma[:max_conv]
            break
        
        # Depending on the linear solver to be used, two mathematically
        # equivalent formulations are used.
        if method == 'pcg':
            # Compute X = X - (P * L * P) \ (P * L * X) where
            # P = I - [e X] * [e X]' is a projection onto the orthogonal
            # complement of [e X].
            W *= Y  # L * X == W * Y
            W -= (W.T * X * X.T).T
            project(W)
            # Compute the diagonal of P * L * P as a Jacobi preconditioner.
            D = L.diagonal().astype(float)
            D += 2. * (asarray(X) * asarray(W)).sum(axis=1)
            D += (asarray(X) * asarray(X * (W.T * X))).sum(axis=1)
            D[D < tol * Lnorm] = 1.
            D = 1. / D
            # Since TraceMIN is globally convergent, the relative residual can
            # be loose.
            #Perform deflation if needed
            W -= X_conv[:,:nconv] * (X_conv[:,:nconv].T * W)
            X -= X_conv[:,:nconv] * (X_conv[:,:nconv].T * X)
            if max_conv:
                X_conv[:,nconv:nconv+max_conv] = X[:,:max_conv]
                sig_conv[nconv:nconv+max_conv] = sigma[:max_conv]
                X = X[:, max_conv:]
                W = W[:, max_conv:]  
                nconv += max_conv
            X -= solver.solve(W, 0.1)
        else:
            # Compute X = L \ X / (X' * (L \ X)). L \ X can have an arbitrary
            # projection on the nullspace of L, which will be eliminated.
            W[:, :] = solver.solve(X)
            project(W)
            X = (inv(W.T * X) * W.T).T  # Preserves Fortran storage order.

    return sig_conv, asarray(X_conv)

def tracemin_solver(L,dim=1, method='pcg', tol=1e-8):
    """
    Find the dim smallest nonzero eigen-values and vectors of L using the tracemin algorithm
    """
    n = L.shape[0]
    q = min(2*dim, L.shape[0]-1)
    X = np.asmatrix(np.random.normal(size=(n, q)))
    #apparently searching a higher dimensional space sometimes gives better results
    normalized = False
    sig, V = _tracemin_fiedler(L, X, normalized, tol, method, num_vecs = dim)
    return sig[:dim], V[:,:dim]

def spectral_embed(G, min_links, dim=1):
    nodes = G.nodes()
    sig, V = tracemin_solver(nx.laplacian_matrix(G), dim=dim)
    points = V/np.sqrt(sig)
    
    pairs, dists = closest_points(points[:,:dim], k=min_links)
    res = []
    for i in range(min_links):
        u, v = nodes[pairs[i][0]],nodes[pairs[i][1]]
        if not G.has_edge(u,v):
            res.append((u,v))
    return res
