#Minimum discrepancy principle strategy for choosing k in kNN regression
#Supplementary functions for knn_estimator_class.py

import numpy as np



def compute_distances_2d(X):
    '''
    Distances between points in data X in the Euclidean norm

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
    Given data

    '''

    dists = np.zeros((len(X), len(X)))

    for i in range(len(X)):
        dists[i, :] = np.sum((X - X[i])**2, axis=1)

    return dists


def fill_matrix_2d(X, k):
    '''
    Returns the matrix of k-nearest neighbors and
    the indices of all neighbors given the 2-dimensional data X

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
    Given data

    k : int,
    The number of nearest neighbors

    '''

    A = np.zeros((len(X), len(X)))
    np.fill_diagonal(A, 1/k)

    dist_matrix = compute_distances_2d(X)
    dist_indices = np.argsort(dist_matrix)
    for i in range(len(X)):
        A[i] = np.zeros(len(X))
        A[i, dist_indices[i, :k]] = 1/k


    return [A, dist_indices]


def compute_distances_cross(X1, X2):
    '''
    Computes the distances and distances' indices in the Euclidean norm
    between two 2-dimensional data: X1 and X2

    Parameters
    ----------
    X1 : array-like of shape (n_samples_1, n_features)
    Data number 1

    X2 : array-like of shape (n_samples_2, n_features)
    Data number 2

    '''
    dists = np.zeros((len(X1), len(X2)))
    for i in range(len(X1)):
        dists[i, :] = np.sum((X2 - X1[i])**2, axis=1)

    dist_indices = np.argsort(dists)
    return [dists, dist_indices]


def fill_matrix_cross(X1, X2, k):
    '''
    Returns the matrix of k-nearest neigbors and
    the indices of all neighbors given two 2-dimensional data: X1 and X2

    Parameters
    ----------
    X1 : array-like of shape (n_samples_1, n_features)
    Data number 1

    X2 : array-like of shape (n_samples_2, n_features)
    Data number 2

    k : int,
    the number of nearest neighbors

    '''
    A = np.zeros((len(X1), len(X2)))

    [dist_matrix, dist_indices] = compute_distances_cross(X1, X2)
    for i in range(len(X1)):
        A[i] = np.zeros(len(X2))
        A[i, dist_indices[i, :k]] = 1/k

    return [A, dist_indices]


def oracle(risk):
    '''
    Computing the oracle stopping rule of the risk function for any estimator.

    Parameters
    ----------
    risk : array-like of shape (max_number_of_estimators,)
    The vector of the risk function of an estimator

    '''

    K = risk.shape[0]

    minim = risk[0]
    minim_k = 1
    for k in range(1, K):
        if (risk[k] < minim):
            minim = risk[k]
            minim_k = k + 1

    return minim_k


def sigma_sq_est(y, A_k, k=2):
    '''
    Estimates the value of the noise level (\sigma^2) in the non-parametric regression model:
    y_i = f^*(x_i) +  \varepsilon_i, i = 1, ..., n, where
    \varepsilon_i follows the zero-mean normal distribution with the variance \sigma^2.

    Parameters
    ----------
    y : array-like of shape (n_samples,)
    The vector of target in the regression model

    A_k : array-like of shape (n_samples, n_samples)
    The matrix of nearest neighbors used for the estimation 

    k : int, default=2,
    The value of the number of nearest neighbors to make the estimation

    '''

    n = len(y)
    return (np.linalg.norm(y - np.dot(A_k, y))**2) / (n * (1 - 1/k))
