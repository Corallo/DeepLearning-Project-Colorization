import torch
from torch.nn import functional as F
from torch.autograd import Variable
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import argparse
from model import NNet
from datasets import ImageNet
import utils
from skimage.color import lab2rgb, rgb2lab
import cv2
import os
from PIL import Image
import torchvision.transforms as transforms


parser = argparse.ArgumentParser(description='Image Colorization')
parser.add_argument('--test_root', default='/dataset/test', type=str)


def main():
    global args, best_prec1
    args = parser.parse_args()
    print(args)
    
    # load the pre-trained weights
    model = NNet()
    # model = torch.nn.DataParallel(model).cuda()
    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load('models/model_latest.pth.tar', map_location=torch.device('cpu'))['state_dict'])
    

    test_dataset = ImageNet(args.test_root)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False,num_workers=8, pin_memory=True)
    print("=> Loaded data, length = ", len(test_dataset))
    model.eval()
    for i, (img, target) in enumerate(test_loader):
        if img is None:
            continue
        # var = Variable(img.float(), requires_grad=True).cuda()
        var = Variable(img.float(), requires_grad=True)
        output = model(var)
        decoded_output = utils.decode(output)
        lab = np.zeros((256,256,3))
        lab[:,:,0] = cv2.resize((img+50.0).squeeze(0).squeeze(0).numpy(), (256,256))
        lab[:,:,1:] = cv2.resize(decoded_output.squeeze(0),(256,256))
        rgb = lab2rgb(lab)
        plt.imshow(rgb)
        plt.savefig("Results/"+ str(i)+'.jpeg')
        print("Forwarded image number: " + str(i+1))


if __name__ == '__main__':
    main()
