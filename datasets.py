from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os
import cv2
from sklearn.neighbors import NearestNeighbors
import numpy as np
from scipy.special import softmax

class ImageNet(Dataset):
    def __init__(self, rootDir):
        self.rootDir = rootDir

        self.transf = transf = transforms.ToTensor()
        # Prepare list data paths
        self.listData = []
        imgPaths = os.listdir(self.rootDir)
        for dirPath in imgPaths:
            totalDirPath = os.path.join(self.rootDir,dirPath)
            imagePaths = os.listdir(totalDirPath)
            for imagePath in imagePaths:
                self.listData.append(os.path.join(totalDirPath,imagePath))

        #ab_bins = np.load('pts_in_hull.npy')
        #nbrs = NearestNeighbors(n_neighbors=5, algorithm='kd_tree', p=2).fit(ab_bins)
        #def soft_encode_i(x):
        #    distances, indices = nbrs.kneighbors(x.reshape((1,-1)))
        #    res = np.zeros(313)
        #    res[indices.reshape(-1)] = softmax(np.exp(-(distances.reshape(-1)**2)/50))
        #    return res

        #self.soft_encode_i = soft_encode_i

    def __len__(self):

        return len(self.listData)

    def __getitem__(self, i):

        imgPath = self.listData[i]
        inputImage = cv2.cvtColor(cv2.imread(imgPath), cv2.COLOR_RGB2LAB)
        image_ab = cv2.resize(inputImage, (64, 64), interpolation = cv2.INTER_AREA)[:,:,1:].astype(float) - 128.0
        image_L = self.transf(inputImage[:,:,0].astype(float))

        #encoded_ab = self.transf(np.apply_along_axis(self.soft_encode_i,-1,image_ab))
        
        return image_L, image_ab