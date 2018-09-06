from sklearn.neighbors import KDTree, BallTree
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist, squareform
from math import ceil
import numpy as np
from collections import Counter

def all_closest(points, k):
    tree = KDTree(points)
    dists,inds = tree.query(points, k+1)
    n = points.shape[0]
    a = np.arange(n).reshape((n,1))
    mask = (inds != a)
    ret_dists = np.empty((n,k))
    ret_inds = np.empty((n,k))
    for i in range(n):
        mask = inds[i] != i
        if sum(mask) > k:
            mask = mask[:k]
        
        ret_inds[i] = inds[i][mask]
        ret_dists[i] = dists[i][mask]
    return ret_dists, ret_inds
    
def brute_closest_points(points, k=1):
    n = len(points)
    dists = pdist(points) #n-1, n-2, ..., n-3
    inds = np.argsort(dists)[:k]

    ind_arr = np.zeros((int(n*(n-1)/2), 2), dtype=int)
    row = 0

    for i in range(n):
        for j in range(i+1,n):
            ind_arr[row] = np.array([i,j])
            row += 1
    
    
    return ind_arr[inds], dists[inds]

def remove_duplicate_points(points, dists):
    points = np.sort(points)
    max_ind = np.max(points, axis=None)
    keys = max_ind * points[:,0] +  points[:,1]
    keys, inds = np.unique(keys, return_index=True)
    return points[inds], dists[inds]

def metric(x,y):
    return np.sqrt(np.sum((x-y)**2))

def closest_points(points, k=1, tol=1e-11):
    """
    Given an array of points in Euclidean space, find the k pairs of points which are closest.
    """
    n = points.shape[0]
    if not n: return np.empty((0,2), dtype=int), np.array([])
    
    l = int(ceil((4*k)/float(n)))
    if n*n <= 4*k or n <= l:
        return brute_closest_points(points,k)
    
    dists, inds = all_closest(points,l)
    npairs =  2*k
    #find the indices corresponding to the closest 2*k distances, allowing repetition
    #These aren't necessarily sorted, except for the very last distance
    #Returns a tuple of row numbers (source points and column numbers (number of neighbor) 
    best_k_inds = np.unravel_index(np.argpartition(dists, npairs, axis=None)[:npairs], dists.shape)
    
    #Use these indices to get the corresponding pairs of points
    pairs = np.empty((npairs, 2), dtype=int)
    pairs[:,0] = best_k_inds[0]
    pairs[:,1] = inds[best_k_inds]

    neighbor_counter = Counter(pairs[:,0])        
    #the distances for these pairs
    pdists = dists[best_k_inds]
    pairs, pdists = remove_duplicate_points(pairs, pdists)
    #filter only the points that have all l nearest neighbors closer than the furthest pair so far
    #we then recursively apply the algorithm to this set of points
    #Figure out how many times each source index is repeated
    mask = np.array([source for source in neighbor_counter if neighbor_counter[source] >= l], dtype=int)                 

    S = points[mask]
    s_closest, s_dists = closest_points(S, k, tol)
    index_list = mask

    firsts = index_list[s_closest[:,0]]
    seconds = index_list[s_closest[:,1]]
    #combine the recursively found best k pairs with the hueristic best 2k pairs
    all_points = np.zeros((len(firsts)+len(pdists),2), dtype = int)
    all_points[:len(pdists)] = pairs
    all_points[len(pdists):] = np.vstack([firsts, seconds]).T

    all_dists = np.concatenate([pdists, s_dists])
    all_points, all_dists = remove_duplicate_points(all_points, all_dists)
    #take the best k pairs out of all
    final_best_k = np.argsort(all_dists)[:k]

    return all_points[final_best_k], all_dists[final_best_k]
    


