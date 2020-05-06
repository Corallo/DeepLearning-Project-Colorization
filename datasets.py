from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os
import cv2
import numpy as np
import torch
from PIL import Image

class ImageNet(Dataset):
    def __init__(self, rootDir):
        self.rootDir = rootDir

        self.transf = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
        ])
        self.toTensor = transforms.ToTensor();
        # Prepare list data paths
        self.listData = []
        imgPaths = os.listdir(self.rootDir)
        for dirPath in imgPaths:
            totalDirPath = os.path.join(self.rootDir,dirPath)
            imagePaths = os.listdir(totalDirPath)
            for imagePath in imagePaths:
                self.listData.append(os.path.join(totalDirPath,imagePath))

        #ab_bins = np.load('pts_in_hull.npy')
        #nbrs = NearestNeighbors(n_neighbors=5, algorithm='ball_tree', p=2).fit(ab_bins)
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
        img = cv2.imread(imgPath)
        if img is None:
            return None, None
        inputImage = np.array(self.transf(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_RGB2LAB))))
        image_ab = self.toTensor(cv2.resize(inputImage, (56, 56), interpolation = cv2.INTER_AREA)[:,:,1:].astype(float) - 128.0)
        image_L = torch.from_numpy(inputImage[:,:,0].astype(float)*100.0/255.0).unsqueeze_(0) - 50.0
        #encoded_ab = self.transf(np.apply_along_axis(self.soft_encode_i,-1,image_ab))
        
        return image_L, image_ab