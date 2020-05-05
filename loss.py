import matplotlib.pyplot as plt 
import numpy as np
from skimage import io, color
import cv2
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
import torch
from torch.nn.functional import softmax

empirical_probs = (0.5*np.load('prior_probs.npy') + (0.5/313))**(-1)
empirical_probs = empirical_probs/np.sum(empirical_probs)

def GaussianKernel(v1, v2, sigma=5):
    return np.exp(-np.linalg.norm(v1-v2, 2)**2/(2.*sigma**2))

def generateRandomImage(h=66,w=66):
    Img = np.random.uniform(-110,110,(h,w,3))
    return Img

#Get in input a LAB image HxWx3. Returns in output his quantized, binary color scheme  Z HxWxQ with Q=313
def getZfromY(nbrs,colorsList,Y,h=66,w=66): 
    #Get only ab channels from image
    pictureColor = Y.reshape(h*w,3)[:,1:]
    distances, indicies = nbrs.kneighbors(pictureColor)
    Z = []
    Q=colorsList.shape[0]
    for pixel, index in zip(pictureColor,indicies):
        dist = []
        for neighbor in colorsList[index]:
            dist.append(GaussianKernel(pixel,neighbor)) #Gaussian distance? (I am a noob, couldn't broadcast)
        dist = dist / np.sum(dist) #Normalize?
        row = np.zeros(Q)
        row[index]= dist
        Z.append(row)
    return np.array(Z).reshape(h,w,Q)

def loadColorData(filename):
    colorsList = np.load(filename)
    nbrs = NearestNeighbors(n_neighbors=5).fit(colorsList)
    return nbrs, colorsList

def v(Z):
    args = torch.argmax(Z,dim=1)
    ant_size = tuple(args.size())
    return torch.from_numpy(empirical_probs[args.cpu().reshape(-1)].reshape(ant_size)).cuda()

def classificationLoss(Z_hat, Z):

    loss = - torch.sum(v(Z) * torch.sum(Z.cuda() * torch.log(softmax(Z_hat, dim=1)),dim=1))
    return loss

def regressorLoss(Z_hat,Z):
    return (1/2)*(numpy.linalg.norm(Z-Z_hat))**2

"""
model,colorsList=loadColorData("pts_in_hull.npy")
randomImg = generateRandomImage()
randomImg2 = generateRandomImage()
z_hat =np.random.uniform(0.1,1,size=(66,66,313))
z = getZfromY(model,colorsList,randomImg)
l=classificationLoss(z_hat,z)

"""
