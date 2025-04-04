# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 11:04:34 2025

@author: Susana Nascimento
The Xie-Beni internal validity index to evaluate fuzzy c-partitions

References:
Pal, N.R. and Bezdek, J.C., 1995. On cluster validity for the fuzzy c-means model. 
                                  IEEE Transactions on Fuzzy systems, 3(3), pp.370-379.    
Xie, X.L. and Beni, G., 1991. A validity measure for fuzzy clustering. 
                              IEEE Transactions on Pattern Analysis & Machine Intelligence, 13(08), pp.841-847.    
"""

import numpy as np

def xie_beni_index(U, centers, X):
    n_clusters, n_samples = U.shape
    compactness = 0.0
    for i in range(n_clusters):
        for j in range(n_samples):
            diff = X[j] - centers[i]
            compactness += (U[i, j] ** 2) * np.dot(diff, diff)
    min_center_dist_sq = np.inf
    for i in range(n_clusters):
        for j in range(i + 1, n_clusters):
            dist_sq = np.sum((centers[i] - centers[j]) ** 2)
            if dist_sq < min_center_dist_sq:
                min_center_dist_sq = dist_sq
    if min_center_dist_sq == 0:
        return np.inf
    return compactness / (n_samples * min_center_dist_sq)


# Optimized Vectorized implementation
def xie_beni_index(U, centers, X):
    um = U ** 2
    dist_sq = np.sum((X[np.newaxis, :, :] - centers[:, np.newaxis, :]) ** 2, axis=2)
    compactness = np.sum(um * dist_sq)

    center_dist_sq = np.sum((centers[:, np.newaxis, :] - centers[np.newaxis, :, :]) ** 2, axis=2)
    np.fill_diagonal(center_dist_sq, np.inf)
    min_center_dist_sq = np.min(center_dist_sq)

    if min_center_dist_sq == 0:
        return np.inf
    return compactness / (X.shape[0] * min_center_dist_sq)
