import torchvision.models as models
from torchvision import datasets, transforms
import torch
from torch.autograd import Variable
import cv2
from PIL import Image
from torchvision import transforms
import json
import os

filepath = 'img/imagenet-mini/generated/'
vgg = models.vgg16(pretrained=True)
vgg.eval()
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
#transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])
#dataset = datasets.ImageNet("D:\Torrent", split="val", transform=transform)
target = []
with open('labels_json.json', 'r') as f:
  labels = json.load(f)
correct=0
total = 0
dir_val = os.listdir(filepath)
print(len(dir_val))
#test_loader = torch.utils.data.DataLoader("D:\Torrent\ILSVRC2012_img_val", batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
for i in range(len(dir_val)):
    print(i)
    imgs_paths = os.listdir(filepath+ dir_val[i])
    for img_path in imgs_paths:

        input_image = Image.open(filepath + dir_val[i] + "/" + img_path)
        if(input_image.mode=='L'): #skip black and white
            continue

        input_tensor = preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

        # move the input and model to GPU for speed if available
        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')
            vgg.to('cuda')

        with torch.no_grad():
            output = vgg(input_batch)
            # Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
        prediction = torch.argmax(output).item()
        target = [key for key, values in labels.items() if values[0] == dir_val[i]]
        target = int(target[0])
        # print(prediction,target)
        correct += (prediction == target)
        total += 1

print("Number of correct classifications: ", correct)
print("Accuracy:", correct/total)
