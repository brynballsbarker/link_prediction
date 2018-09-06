from scipy.sparse import linalg as sla
import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix
import scipy.linalg as la

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
    return np.array([item for item in first if item not in second])
    
def get_loc(pos, n):
    """
    takes position in flattened array and gives location in non-flattened
    """
    row = int(pos/n)
    col = pos % n
    return (row, col)
    
def partition(M, r, c):
    """
    returns the r rows and c columns of sparse matrix M
    """
    part = M[:,c]
    part = part[r]
    return part
    
def scipy_eigsh(M, dim=1, tol=1e-8):
    """
    returns the eigenvalue of largest magnitude corresponding to matrix M
    """
    M = M.astype(np.float64)
    sigma = sla.eigsh(M, k=dim, which='LM', tol=tol, return_eigenvectors=False)
    return sigma[0]
    
def project_space(space, output_dim):
	"""
	project given space onto one of dim output_dim, using SVD
	"""
	U, s, V = la.svd(space)
	if output_dim > s.size:
		print("This is a problem")
		output_dim = s.size
	s = s[:output_dim]
	U = U[:,:output_dim]
	for i in range(output_dim):
		U[:,i] = U[:,i]*s[i]
	return U
    
def smallest_laplacian_eigenspaces(L, num_spaces=1, force_dim=None, tol=1e-10, return_evals=False, **kwargs):
	"""
	num_spaces=1: same as fiedler_eigenspace
	else: returns nxk 
	"""
	evals, evecs = la.eigh(L)
	inds = np.argsort(evals)
	spaces = 0
	i = 1
	n = evals.size
	evals = np.sort(evals)
	res = None
	
	if force_dim is not None:
		num_spaces = n 
	
	while i<n and spaces<num_spaces:
		curr_eval = evals[i]
		spaces += 1
		if i == 1:
			res = evecs[:,i].reshape(n,1)
		else:
			res = np.concatenate((res,evecs[:,i].reshape(n,1)),axis=1)
		while i<n:
			i+=1
			if i<n and abs(evals[i]-curr_eval)<tol:
				res = np.concatenate((res,evecs[:,i].reshape(n,1)),axis=1)
			else:
				break
		
		if force_dim is not None:
			curr_dim = res.shape[1]
			if force_dim <= curr_dim:
				if not force_dim==curr_dim:
					print("Projecting")
					res = project_space(res, force_dim)
					break
				else:
					break
	
	if return_evals:
		return res, evals[1:res.shape[1]+1]
	return res 
			
		
		
		
	
	
	
	
	
	
	
	
	
	
