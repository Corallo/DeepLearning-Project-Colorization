import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.special import softmax
import cv2
import matplotlib.pyplot as plt
from skimage import color
from skimage.color import rgb2lab, rgb2gray
from PIL import Image
#import torch

ab_bins = np.load('pts_in_hull.npy')
nbrs = NearestNeighbors(n_neighbors=5, algorithm='kd_tree', p=2).fit(ab_bins)

def soft_encode_ab(raw_ab):

    #raw_ab = raw_ab.numpy()

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
    
    return encoded_ab

def getYgivenZ(Z):

    Y=np.argmax(Z,axis=-1)
    ab_img=ab_bins[Y[:][:]]

    # Z=Z.reshape((-1,Q))
    # num = np.exp(np.log(Z)/T)
    # den = np.sum(np.exp(np.log(Z)/T),axis=1)
    # ft= num/den[:,None]
    # assert(np.sum(ft,axis=0).all(0)==1) #should sum 1
    # Y=np.dot(ft,colorsList).reshape(w,h,2)
    # return Y

    return ab_img

def testfun():
    #img = np.random.randint(0,255,(64,64,3),dtype=np.uint8)
    imgPath = "img/img.jpeg"
    inputImage = cv2.cvtColor(cv2.imread(imgPath), cv2.COLOR_RGB2LAB)
    raw_ab = cv2.resize(inputImage, (64, 64), interpolation = cv2.INTER_AREA)[:,:,1:].astype(float) - 128.0
    raw_ab=raw_ab.transpose([2,0,1])
    new_ab = np.empty([1,2,64,64])
    new_ab[0,:,:,:]= raw_ab
    
    Z = soft_encode_ab(new_ab)
    Z = Z[0,:,:,:].transpose([1,2,0])

    ab_img = getYgivenZ(Z)

    light = np.empty((64,64,1))
    light[:,:,0]= cv2.resize(inputImage[:,:,0], (64,64))
    lab_img= np.concatenate((light, ab_img), axis=-1)
    rgb = color.lab2rgb(lab_img)

    plt.imshow(rgb)
    plt.show()

testfun()