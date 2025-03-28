

## TODO é preciso mudar codigo!!!! renomear varaiaveis para correr com o data set dado



# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 23:23:59 2020

@author: Susana Nascimento
"""
import numpy as np
# Pandas also for data management
import pandas as pd

## Anomalous Pattern Algorithm

def center_(x, cluster):
    """ 
    calculates the centroid of the cluster
    x - the original data matrix ( N x d)
    cluster - the set with indices (i= 1, 2, ..., N) of the objects belonging to the cluster
    returns the centroid of the cluster 
    """
    #number of columns
    mm = x.shape[1]
    centroidC = []
    
    for j in range(mm):
        zz = x[:, j]
        zc = []
        for i in cluster:
            zc.append(zz[i])
        centroidC.append(np.mean(zc))
    return centroidC


def distNorm(x ,remains, ranges, p):
    """ 
     Finds the normalized distances of data points in 'remains' to reference point 'p' 
     x - the original data matrix ( N x d)
     remains- the set of X-row indices: the indices of the entities under consideration
     ranges- the vector with the ranges of the data features  
     p - the reference data point the distances relate to
     distan- returns the column vetor  with the distances from p to remains 
     """

    
    mm = x.shape[1]      # number of data features
    rr = len(remains)    # number of entities in remains    
    z = x[remains, :]
    az = np.tile(np.array(p), (rr, 1))     # Construct an array by repeating input array np.array(p)  
                                           # the number of rows is rr
    
    rz = np.tile(np.array(ranges), (rr, 1))
    dz = (z - az) / rz
    dz = np.array(dz)
    ddz = dz * dz
    if mm > 1:
        di = sum(ddz.T)
    else:
        di = ddz.T
    distan = di
    return distan


def separCluster(x0, remains, ranges, a, b):
    """  
    Builds a cluster by splitting the points around the reference point 'a' from those around the reference point b 
    x0 - data matrix
    remains- the set of X-row indices: the indices of the entities under consideration
    ranges-  the vector with the ranges of the data features  
    a, b - the reference points
    cluster - returns a set with the row indices of the objects belonging to the cluster  
    """
    
    dista = distNorm(x0, remains, ranges, a)
    distb = distNorm(x0, remains, ranges, b)
    clus = np.where(dista < distb)[0]
    cluster = []
    for i in clus:
        cluster.append(remains[i])
    return cluster


 ## Consult description of building an Anomalous cluster (lecture K-means clustering)
def anomalousPattern(x, remains, ranges, centroid, me):
    """ Builds one anomalous cluster based on the algorithm 'Separate/Conquer' (B. Mirkin (1999): Machine Learning Journal) 
        x - data matrix,
        remains - the set of X-row indices: the indices of the entities under consideration
        ranges - normalizing values: the vector with ranges of data features  
        centroid - initial center of the anomalous cluster being build
        me - vector to shift the 0 (origin) to
        Returns a tuple with:
                cluster - set of (remains) row indices in the anomalous cluster, 
                centroid - center of the built anomalous cluster    
    """        
    key = 1
    while key == 1:
        cluster = separCluster(x, remains, ranges, centroid, me)
        if len(cluster) != 0:
            newcenter = center_(x, cluster)
          
        if  len([i for i, j in zip(centroid, newcenter) if i == j]) != len(centroid):
            centroid = newcenter
        else:
            key = 0
    return (cluster, centroid)

def dist(x, remains, ranges, p):
    """ 
      Calculates the normalized distances of data points in 'remains' to reference point 'p'   
       x - data matrix,
       remains - the set of X-row indices: the indices of the entities under consideration
       ranges - normalizing values: the vector with ranges of data features      
       distan - returns the calculated normalized distances
    """

    
    mm = x.shape[1]       #number of columns
    rr = len(remains)     # number of entities in remains  
    distan = np.zeros((rr,1))    
    for j in range(mm):
        z = x[:, j]         # j feature vector
        z = z.reshape((-1,1))
        zz = z[remains]
        y = zz - p[j]
        y = y / ranges[j]
        y = np.array(y)
        yy = y * y
        distan = distan + yy
    return distan



##### ****** Main body for the Iterative Anomalous Cluster Algorithm  *****
#### You should test and Validate it with the Market Towns Data set following the report 

# normalization FLAG
normalization = 0
# threshold value of the cardinality of clusters (this is an example): 
threshold = 25

### I consider this from the PCA transformation
### Must be explored and adapted concerning the best normalization for specific data set


data_ap =  zscor_data_pca.iloc[:,:-1]     # from pandas dataframe TODO
x = data_ap.values.astype(np.float32)
#y = data.target

#number of data points
nn = x.shape[0]
#number of features
mm = x.shape[1]

# grand means
me = []
# maximum value
mmax = []
# minimum value
mmin = []
# ranges
ranges = []
# "anomalous cluster" ancl is the data structure to keep everything together
ancl = []


for j in range(mm): # for each feature
    z = x[:, j]     # data column vector j-th feature
    me.append(np.mean(z))
    mmax.append(np.max(z))
    mmin.append(np.min(z))
    if normalization:
        ranges.append(1);
    else:
        ranges.append(mmax[j] - mmin[j])
    if ranges[j] == 0:
        print("Variable num {} is contant!".format(j))
        ranges[j] = 1
    
sy = np.divide((x - me), ranges)
sY = np.array(sy)
d = np.sum(sY * sY)   # total data scatter of normalized data


# x, me range, d
remains = list(range(nn))  # current index set of residual data after some anomalous clusters are extracted
numberC = 0; # counter of anomalous clusters 
while(len(remains) != 0):
    distance = dist(x, remains, ranges, me) # finding normalised distance vector from remains data points to reference 'me'
    ind = np.argmax(distance)
    index = remains[ind]
    centroid = x[index, :]   # initial anomalous center reference point: the one with higher distance
    numberC = numberC + 1
    
    (cluster, centroid) = anomalousPattern(x, remains, ranges, centroid, me) # finding AP cluster
    
    
    censtand = np.divide((np.asarray(centroid) - me), np.asarray(ranges)) # standardised centroid with parameters of the data   
    dD = np.sum(np.divide(censtand * censtand.T * len(cluster) * 100, d)) # cluster contribution (per cent) - (lecture on K-means and iK-means)

    remains = np.setdiff1d(remains, cluster) 
    # update the data structure that keeps everything together
    ancl.append(cluster)   # set of data points in the cluster
    ancl.append(censtand)  # standardised centroid
    ancl.append(dD) # proportion of the data scatter
    
ancl = np.asarray(ancl)         # convert to an array
ancl = ancl.reshape((numberC, 3))
##aK = numberC
b = 3
ll = [] # list of clusters

for ik in range(numberC):
    ll.append(len(ancl[ik, 0]))
    
rl = [i for i in ll if i >= threshold] # list of clusters with at least threshold elements
cent = []
if(len(rl) == 0):
    print('Too great a threhsold!!!')
else:
    num_cents = 0
    for ik in range(numberC):
        cluster = ancl[ik,0]
        if(len(cluster) >= threshold):
            cent.append(ancl[ik, 1])
            num_cents += 1
                
cent = np.asarray(cent)

### Should be adapted
#cent = cent.reshape((len(cent), len(zscor_data_pca.columns) - 1))
##print("Initial prototypes: \n", np.round(cent, DECIMAL_PLACES))
#init_partition = np.zeros((zscor_data_pca.shape[0], len(cent)))

#for index, d in enumerate(zscor_data):
#    dists = [np.linalg.norm(d - c) for c in cent]
#    assign = dists.index(np.min(dists))
#    init_partition[index, assign] = 1
    
