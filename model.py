import torch


class NNet(torch.nn.Module):
    # CNN class holder
    # In pytorch, ANNs inherit from torch.nn.Module class

    def __init__(self, regr=True):
        super(NNet, self).__init__()
        # Here we define the network architecture.
        # Following https://arxiv.org/pdf/1603.08511.pdf,
        # the network is divided in 8 convolutional blocks (conv1-8)
        # and one last 313 feature output block
        
        # We use the nn.Sequential() method to represent all the layers
        # in each convolutional block
        self.regr = regr
        #  --- conv1 --- 
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3,
                stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3,
                stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(num_features=64, affine=False)
        )

        #  --- conv2 --- 
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3,
                stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3,
                stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(num_features=128, affine=False)
        )

        #  --- conv3 --- 
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3,
                stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3,
                stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3,
                stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(num_features=256, affine=False)
        )

        #  --- conv4 --- 
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3,
                stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3,
                stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3,
                stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(num_features=512, affine=False)
        )

        #  --- conv5 --- 
        self.conv5 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3,
                stride=1, padding=2, dilation=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3,
                stride=1, padding=2, dilation=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3,
                stride=1, padding=2, dilation=2),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(num_features=512, affine=False)
        )

        #  --- conv6 --- identical to conv5 block
        self.conv6 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3,
                stride=1, padding=2, dilation=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3,
                stride=1, padding=2, dilation=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3,
                stride=1, padding=2, dilation=2),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(num_features=512, affine=False)
        )

        #  --- conv7 --- 
        self.conv7 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3,
                stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3,
                stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3,
                stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(num_features=512, affine=False)
        )

        #  --- conv8 --- 
        self.conv8 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4,
                stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3,
                stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3,
                stride=1, padding=1),
            torch.nn.ReLU()
        )

        # 1x1 Cross Entropy Loss
        self.conv_ab = torch.nn.Conv2d(in_channels=256, out_channels=313, kernel_size=1,
                stride=1, padding=0)
        #Regressor Loss
        self.regressor = torch.nn.Conv2d(in_channels=256, out_channels=2, kernel_size=1,
                stride=1, padding=0)
    def forward(self, image):

        # Simple forward pass to the network
        # through all the convolutional blocks

        conv1 = self.conv1(image)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        conv6 = self.conv6(conv5)
        conv7 = self.conv7(conv6)
        conv8 = self.conv8(conv7)

        if self.regr:
            return self.regressor(conv8)
        else:
            conv_ab = self.conv_ab(conv8)
            return conv_ab


class DCGAN(torch.nn.Module):
    def __init__(self):
        super(DCGAN, self).__init__()


        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=4, 
                stride=2, padding=1),
            torch.nn.LeakyReLU(0.2, True)
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, 
                stride=2, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.LeakyReLU(0.2, True)
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, 
                stride=2, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.LeakyReLU(0.2, True)
        )
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, 
                stride=2, padding=1),
            torch.nn.BatchNorm2d(512),
            torch.nn.LeakyReLU(0.2, True)
        )
        self.out = torch.nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, 
                stride=1, padding=1)

    def set_grads(self, grads):
        for param in self.parameters():
            param.requires_grad = grads

    def forward(self, image):
        conv1 = self.conv1(image)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        out = self.out(conv4)

        return out
