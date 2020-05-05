import numpy as np
from sklearn.neighbors import NearestNeighbors
import torch

ab_bins = np.load('pts_in_hull.npy')
nbrs = NearestNeighbors(n_neighbors=5, algorithm='kd_tree', p=2).fit(ab_bins)

def soft_encode_ab(raw_ab):

    raw_ab = raw_ab.numpy()

    # Flatten (C, A, H, W) array into (C*H*W, A) array

    nax = np.setdiff1d(np.arange(0,raw_ab.ndim),np.array((1)))
    axorder = np.concatenate((nax,np.array(1).flatten()),axis=0)

    flat_ab = raw_ab.transpose((axorder)).reshape((-1,2))

    # Calculate encoidings for each element

    distances, indices = nbrs.kneighbors(flat_ab)

    dist_w = np.exp(-distances**2/(2*5**2))
    dist_w = dist_w/np.sum(dist_w,axis=1, keepdims=True)

    encoded_ab_flat = np.zeros((flat_ab.shape[0],ab_bins.shape[0]))
    encoded_ab_flat[np.arange(flat_ab.shape[0])[:,None], indices] = dist_w
    
    # Unflatten (C*H*W, Q) array into (C, Q, H, W)

    reversed_ax = np.argsort(axorder)

    enc_shape = np.array(raw_ab.shape)[nax].tolist()
    enc_shape.append(encoded_ab_flat.shape[1])
    encoded_ab = encoded_ab_flat.reshape(enc_shape).transpose(reversed_ax)
    
    return torch.from_numpy(encoded_ab)


def getYgivenZ(Z, w=66, h=66, Q=313, T=0.38, filename="pts_in_hull.npy"):
    colorsList = np.load(filename)
    Z=Z.reshape(w*h,Q)
    num = np.exp(np.log(Z)/T)
    den = np.sum(np.exp(np.log(Z)/T),axis=1)
    ft= num/den[:,None]
    assert(np.sum(ft,axis=0).all(0)==1) #should sum 1
    Y=np.dot(ft,colorsList).reshape(w,h,2)
    return Y
