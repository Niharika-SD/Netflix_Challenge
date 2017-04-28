from __future__ import division, print_function

__all__ = ["LRMC"]

import time
import fbpca
import logging
import numpy as np
from scipy.sparse.linalg import svds

def lrmc(X,method , delta=1e-6, mu=None, maxiter=500, verbose=False,svd_method="approximate", **svd_args):
    # Check the SVD method.
    allowed_methods = ["approximate", "exact", "sparse"]
    if svd_method not in allowed_methods:
        raise ValueError("'svd_method' must be one of: {0}"
                         .format(allowed_methods))

    # Check for missing data.
    shape = X.shape
    if missing_data:
        missing = ~(np.isfinite(X))
        if np.any(missing):
            X = np.array(X)
            M[missing] = 0.0
    else:
        missing = np.zeros_like(X, dtype=bool)
        if not np.all(np.isfinite(X)):
            logging.warn("The matrix has non-finite entries. "
                         "SVD will probably fail.")
    
    W = np.asarray(X ==0,dtype = np.int32)

    # Initialize the tuning parameters.
    lam = 1.2*numel(X)/sum(sum(W))
    
    if mu is None:
        mu = 10*5* sqrt(numel(X))
        if verbose:
            print("mu = {0}".format(mu))

    # Convergence criterion.
    norm = np.sum(X ** 2)

    # Iterate.
    i = 0
    rank = np.min(shape)
    Z = np.zeros(shape)
    A = np.zeros(shape)
    while i < max(maxiter, 1):
        # SVD step.
        Z_dash = Z*W;
        strt = time.time()
        u, s, v = _svd(svd_method, Z_dash, rank+1, 1./mu, **svd_args)
        svd_time = time.time() - strt
    	s = shrink(s,mu)
        rank = np.sum(s > 0.0)
        Z_prev = Z
        u, s, v = u[:, :rank], s[:rank], v[:rank, :]
        A = np.dot(u, np.dot(np.diag(s), v))
    	Z = Z + lam*(W*X -W*A);

	err = (np.sqrt(np.sum((Z-Z_prev)** 2))) / (np.sqrt(np.sum(W**2)))
		if verbose:
			print(("Iteration {0}: error={1:.3e}, rank={2:d}, nnz={3:d}, "
                   "time={4:.3e}")
                  .format(i, err, np.sum(s > 0), np.sum(S > 0), svd_time))
		if err < delta:
			break
		i += 1

	if i >= maxiter:
		logging.warn("convergence not reached in pcp")
	
	return A, W, (u, s, v)


	def shrink(M, tau):
		sgn = np.sign(M)	
		S = np.abs(M) - tau
		S[S < 0.0] = 0.0
		return sgn * S


    def _svd(method, X, rank, tol, **args):
		rank = min(rank, np.min(X.shape))
		if method == "approximate":
			return fbpca.pca(X, k=rank, raw=True, **args)
    	elif method == "exact":
        	return np.linalg.svd(X, full_matrices=False, **args)
		elif method == "sparse":	
			if rank >= np.min(X.shape):
				return np.linalg.svd(X, full_matrices=False)
			u, s, v = svds(X, k=rank, tol=tol)
			u, s, v = u[:, ::-1], s[::-1], v[::-1, :]
        	return u, s, v
		raise ValueError("invalid SVD method")