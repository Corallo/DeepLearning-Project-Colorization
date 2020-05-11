import torchvision.models as models
from torchvision import datasets, transforms
import torch
from torch.autograd import Variable
import cv2
from PIL import Image
from torchvision import transforms

filepath = 'D:\Torrent\ILSVRC2012_img_val'
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
with open(filepath+'\ILSVRC2012_validation_ground_truth.txt', 'r') as f:
  for line in f:
      num = int(line)
      target.append(num)
correct=0
num_test=1000
#test_loader = torch.utils.data.DataLoader("D:\Torrent\ILSVRC2012_img_val", batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
for i in range(num_test):
    filename = filepath+"\ILSVRC2012_val_"+ '{0:08d}.JPEG'.format(i+1)
    #print("Reading "+filename)

    input_image = Image.open(filename)
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
    #print(output[0])
    # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
    #print(torch.nn.functional.softmax(output[0], dim=0))
    prediction = torch.argmax(output).item()
    #print(prediction,target[i])
    correct += (prediction == target[i])

print("Number of correct classifications: ", correct)
print("Accuracy:", correct/num_test)