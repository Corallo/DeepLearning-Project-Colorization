import torch


class NNet(torch.nn.Module):
    # CNN class holder
    # In pytorch, ANNs inherit from torch.nn.Module class

    def __init__(self):
        super(NNet, self).__init__()
        # Here we define the network architecture.
        # Following https://arxiv.org/pdf/1603.08511.pdf,
        # the network is divided in 8 convolutional blocks (conv1-8)
        # and one last 313 feature output block
        
        # We use the nn.Sequential() method to represent all the layers
        # in each convolutional block
        
        #  --- conv1 --- 
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3,
                stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3,
                stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(num_features=64)
        )

        #  --- conv2 --- 
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3,
                stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3,
                stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(num_features=128)
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
            torch.nn.BatchNorm2d(num_features=256)
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
            torch.nn.BatchNorm2d(num_features=512)
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
            torch.nn.BatchNorm2d(num_features=512)
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
            torch.nn.BatchNorm2d(num_features=512)
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
            torch.nn.BatchNorm2d(num_features=512)
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
        conv_ab = self.conv_ab(conv8)

        return conv_ab

class NNetReduced(torch.nn.Module):

    def __init__(self):
        super(NNetReduced, self).__init__()

                #  --- conv1 --- 
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3,
                stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3,
                stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(num_features=64)
        )

        #  --- conv2 --- 
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3,
                stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3,
                stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(num_features=128)
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
            torch.nn.BatchNorm2d(num_features=256)
        )

        #  --- conv5 --- 
        self.conv5 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3,
                stride=1, padding=2, dilation=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3,
                stride=1, padding=2, dilation=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3,
                stride=1, padding=2, dilation=2),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(num_features=512)
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

    def forward(self, image):

        # Simple forward pass to the network
        # through all the convolutional blocks

        conv1 = self.conv1(image)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv5 = self.conv5(conv3)
        conv8 = self.conv8(conv5)
        conv_ab = self.conv_ab(conv8)

        return conv_ab