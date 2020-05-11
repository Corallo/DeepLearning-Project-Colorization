from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os
import numpy as np
import torch
from PIL import Image
import cv2
from skimage.color import rgb2lab

class ImageNet(Dataset):
    def __init__(self, rootDir, output_full=False):
        self.rootDir = rootDir
        self.output_full = output_full
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
                if imagePath.split('.')[-1].lower() != 'jpeg' or imagePath.startswith('.'):
                    print("Found corrupted path", imagePath)
                else:
                    self.listData.append(os.path.join(totalDirPath,imagePath))

    def __len__(self):

        return len(self.listData)

    def __getitem__(self, i):

        imgPath = self.listData[i]
        img = Image.open(imgPath).convert('RGB')

        inputImage = rgb2lab(np.array(self.transf(img))).astype(float)
        inputImage[:,:,0] -= 50.0
        image_ab = self.toTensor(cv2.resize(inputImage, (56, 56), interpolation = cv2.INTER_AREA)[:,:,1:].astype(float))
        image_L = torch.from_numpy(inputImage[:,:,0]).unsqueeze_(0)
        if self.output_full:
            img_all = torch.from_numpy(inputImage)
            return img_all, image_L, image_ab
        else:
            return image_L, image_ab
