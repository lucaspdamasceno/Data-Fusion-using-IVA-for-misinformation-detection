#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Imports

import numpy  as np
import scipy as sc
from sklearn.covariance import GraphicalLassoCV


from helpers_iva import normalize_column_vectors, decouple_trick, bss_isi, whiten_data
from initializations import jbss_sos, cca


def iva_spice(X, whiten=True, verbose=False, A=None, max_iter=512, W_diff_stop=1e-6, alpha0=1.0):
    
    """
    Implementation of all the second-order (Gaussian) independent vector analysis (IVA) algorithms.
    Real valued using gradient optimization.
    
    For a general description of the algorithm and its relationship with others,
    see https://urldefense.com/v3/__http://mlsp.umbc.edu/jointBSS_introduction.html__;!!IaT_gp1N!gcLMVNHaocM9dxMD8uC66Ees6BVgmALY_-uO7NJnnjvczOwdkhMkEEEWmdbo8Yqt2_o$ 


    Parameters
    ----------
    X : np.ndarray
        data matrix of dimensions N x T x K.
        Data observations are from K data sets, i.e., X[k] = A[k] @ S[k], where A[k] is an N x N
        unknown invertible mixing matrix, and S[k] is N x T matrix with the nth row corresponding to
        T samples of the nth source in the kth dataset. This enforces the assumption of an equal
        number of samples in each dataset.
        For IVA, it is assumed that a source is statistically independent of all the other sources
        within the dataset and exactly dependent on at most one source in each of the other
        datasets.

    whiten : bool, optional
        if True, data is whitened.

    verbose : bool, optional
        enables print statements if True. Automatically set to True if A is provided

    A : np.ndarray, optional
        true mixing matrices A of dimensions N x N x K, automatically sets verbose to True

    max_iter : int, optional
        max # of iterations

    W_diff_stop : float, optional
        stopping criterion

    alpha0 : float, optional
        initial step size scaling


    Returns
    -------
    W : np.ndarray
        the estimated demixing matrices of dimensions N x N x K so that ideally
        W[k] @ A[k] = P @ D[k], where P is any arbitrary permutation matrix and D[k] is any
        diagonal invertible (scaling) matrix. Note that P is common to all datasets. This is
        to indicate that the local permutation ambiguity between dependent sources
        across datasets should ideally be resolved by IVA.

    cost : float
        the cost for each iteration

    Sigma_N : np.ndarray
        Covariance matrix of each source component vector, with dimensions K x K x N

    isi : float
        joint isi (inter-symbol-interference) for each iteration. Only available if user
        supplies true mixing matrices for computing a performance metric. Else returns np.nan.


    # TODO: Notes, References

    """
    
    # throw errors for incorrect X argument
    if X.ndim != 3:
        raise AssertionError('X must have dimensions N x T x K.')
    elif X.shape[2] == 1:
        raise AssertionError('There must be ast least K=2 datasets.')

    # get dimensions
    N, T, K = X.shape

    # determine if (correct) A argument provided
    if A is not None:
        supply_A = True
        if A.shape[0] != N or A.shape[1] != N or A.shape[2] != K:
            raise AssertionError('A must have dimensions N x N x K.')
    else:
        supply_A = False

    blowup = 1e3
    alpha_scale = 0.9   # alpha0 to alpha0*alpha_scale when cost does not decrease
    alpha_min = W_diff_stop  # can also be chosen different
        
    if whiten:
        X, V, _ = whiten_data(X)

    # calculate cross-covariance matrices of X
    R_xx = np.zeros((N, N, K, K), dtype=X.dtype)
    for k1 in range(K):
        for k2 in range(k1, K):
            R_xx[:, :, k1, k2] = 1 / T * X[:, :, k1] @ np.conj(X[:, :, k2].T)
            R_xx[:, :, k2, k1] = np.conj(R_xx[:, :, k1, k2].T)  # R_xx is Hermitian

    # Initializations

    # Initialize W through random initialization
    W = np.random.randn(N, N, K)
    Y = np.zeros((N, T, K))
    
    for k in range(K):
        W[:, :, k] = np.linalg.solve(sc.linalg.sqrtm(W[:, :, k] @ W[:, :, k].T), W[:, :, k])
        
    if supply_A:
        # only reason to supply A matrices is to display running performance and to compute output_isi
        verbose = True
        isi = np.zeros(max_iter)
        if whiten:
            # A matrix is conditioned by V if data is whitened
            A_w = np.copy(A)
            for k in range(K):
                A_w[:, :, k] = V[:, :, k] @ A_w[:, :, k]
        else:
            A_w = np.copy(A)
    else:
        isi = np.nan
        
    # Check rank of data-covariance matrix: should be full rank, if not we inflate (this is ad hoc)
    # concatenate all covariance matrices in a big matrix
    R_xx_all = np.moveaxis(R_xx, [0, 1, 2, 3], [0, 2, 1, 3]).reshape(
        (R_xx.shape[0] * R_xx.shape[2], R_xx.shape[1] * R_xx.shape[3]), order='F')
    rank = np.linalg.matrix_rank(R_xx_all)
    if rank < (N * K):
        # inflate Rx
        _, k, _ = np.linalg.svd(R_xx_all)
        R_xx_all += k[rank - 1] * np.eye(N * K)  # add smallest singular value to main diagonal
        R_xx = np.moveaxis(
            R_xx_all.reshape(R_xx.shape[0], R_xx.shape[2], R_xx.shape[1], R_xx.shape[3], order='F'),
            [0, 2, 1, 3], [0, 1, 2, 3])

    # Initialize some local variables
    cost = np.zeros(max_iter)
    cost_const = K * np.log(2 * np.pi * np.exp(1))  # local constant   
    
    grad = np.zeros((N, K))

    # Main Iteration Loop
    for iteration in range(max_iter):
        term_criterion = 0

        # Some additional computations of performance via ISI when true A is supplied
        if supply_A:
            avg_isi, joint_isi = bss_isi(W, A_w)
            isi[iteration] = joint_isi

        W_old = np.copy(W)
        cost[iteration] = 0
        for k in range(K):
            cost[iteration] -= np.log(np.abs(np.linalg.det(W[:, :, k])))
            # Min. cost function

        Q = 0
        R = 0
                
        # Calculate SCV (new code)
        for k in range(K):
            Y[:,:,k] = np.matmul(W[:,:,k], X[:,:,k])
        
        # Loop over each SCV
        for n in range(N):
            Wn = np.conj(W[n, :, :]).flatten(order='F')
                   
            Sigma_n = np.eye(K)
            
            # Graphical Lasso - SPICE (new code)
            model = GraphicalLassoCV(tol=1e-2, max_iter=200) # max_iterations increase
            model.fit(Y[n, :, :])
            Sigma_n = model.covariance_
            Sigma_inv = model.precision_
                        
            cost[iteration] += 0.5 * (cost_const + np.log(np.abs(np.linalg.det(Sigma_n))))

            # Decoupling trick
            hnk, Q, R = decouple_trick(W, n, Q, R)

            for k in range(K):
                # Analytic derivative of cost function with respect to vn
                # Code below is efficient implementation of computing the gradient, which is
                # independent of T
                grad[:, k] = - hnk[:, k] / (W[n, :, k] @ hnk[:, k])


                for kk in range(K):
                    grad[:, k] += R_xx[:, :, k, kk] @ np.conj(W[n, :, kk]) * Sigma_inv[kk, k]

                wnk = np.conj(W[n, :, k])

                grad_norm = normalize_column_vectors(grad[:, k])
                grad_norm_proj = normalize_column_vectors(grad_norm - np.conj(
                    wnk) @ grad_norm * wnk)  # non-colinear direction normalized
                W[n, :, k] = np.conj(
                    normalize_column_vectors(wnk - alpha0 * grad_norm_proj))

        for k in range(K):
            # in original matlab code, this is the term criterion
            term_criterion = np.maximum(term_criterion, np.amax(
                1 - np.abs(np.diag(W_old[:, :, k] @ np.conj(W[:, :, k].T)))))

        # Decrease step size alpha if cost increased from last iteration
        if iteration > 0:
            if cost[iteration] > cost[iteration - 1]:
                alpha0 = np.maximum(alpha_min, alpha_scale * alpha0)

        # Check the termination condition
        if term_criterion < W_diff_stop or iteration == max_iter - 1:
            break
        elif term_criterion > blowup or np.isnan(cost[iteration]):
            for k in range(K):
                W[:, :, k] = np.eye(N) + 0.1 * np.random.randn(N, N)
            if verbose:
                print('W blowup, restart with new initial value.')

        # Display Iteration Information
        if verbose:
            if supply_A:
                print(
                    f'Step {iteration}: W change: {term_criterion}, Cost: {cost[iteration]}, '
                    f'Avg ISI: {avg_isi}, Joint ISI: {joint_isi}')
            else:
                print(f'Step {iteration}: W change: {term_criterion}, Cost: {cost[iteration]}')

    # Finish Display
    if iteration == 0 and verbose:
        if supply_A:
            print(
                f'Step {iteration}: W change: {term_criterion}, Cost: {cost[iteration]}, '
                f'Avg ISI: {avg_isi}, Joint ISI: {joint_isi}')
        else:
            print(f'Step {iteration}: W change: {term_criterion}, Cost: {cost[iteration]}')

    # Clean-up Outputs
    cost = cost[0:iteration + 1]

    if supply_A:
        isi = isi[0:iteration + 1]

    if whiten:
        for k in range(K):
            W[:, :, k] = W[:, :, k] @ V[:, :, k]
    else:  # no prewhitening
        # Scale demixing vectors to generate unit variance sources
        for n in range(N):
            for k in range(K):
                W[n, :, k] /= np.sqrt(W[n, :, k] @ R_xx[:, :, k, k] @ np.conj(W[n, :, k]))

    # Resort order of SCVs: Order the components from most to least ill-conditioned

    # First, compute the data covariance matrices (by undoing any whitening)
    if whiten:
        for k1 in range(K):
            for k2 in range(k1, K):
                R_xx[:, :, k1, k2] = np.linalg.solve(V[:, :, k2],
                                                     np.linalg.solve(V[:, :, k1],
                                                                     R_xx[:, :, k1, k2]).T).T
                R_xx[:, :, k2, k1] = np.conj(R_xx[:, :, k1, k2].T)  # R_xx is Hermitian

    # Second, compute the determinant of the SCVs
    detSCV = np.zeros(N)
    Sigma_N = np.zeros((K, K, N))
    
    for k in range(K):
        Y[:,:,k] = np.matmul(W[:,:,k], X[:,:,k])

    for n in range(N):
        
        Sigma_n = np.zeros((K, K))
        
        # Graphical Lasso - SPICE (new code)
        model = GraphicalLassoCV(tol=1e-2, max_iter=200) # max_iterations increase
        model.fit(Y[n, :, :])
        Sigma_n = model.covariance_
        Sigma_inv = model.precision_
                
        Sigma_N[:, :, n] = Sigma_n

        detSCV[n] = np.linalg.det(Sigma_n)

    # Third, sort and apply
    isort = np.argsort(detSCV)
    Sigma_N = Sigma_N[:, :, isort]
    for k in range(K):
        W[:, :, k] = W[isort, :, k]

    return W, cost, Sigma_N, isi    

