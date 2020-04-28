from skimage import data
from skimage.transform import resize
import torch
from torch.autograd import Variable
from model import NNet
from matplotlib import pyplot as plt

nnet = NNet()

img = data.camera()
img = torch.from_numpy(resize(img, (256, 256))).float()

var = Variable(img, requires_grad=True).unsqueeze(0).unsqueeze(0)

result = nnet(var)

print(result.shape)







