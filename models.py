# Import the required libraries
import torch
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import torch.nn as nn

class Model1(nn.Module):
    def __init__(self):
        super(Model1, self).__init__()
        self.down_layers = nn.Sequential(
            self.__downBlock(3, 6, 3, 1, 1, 2, 2),   #Output is 6x256x256
            self.__downBlock(6, 12, 3, 1, 1, 2, 2),  #Output is 12x128x128
            self.__downBlock(12, 32, 3, 1, 1, 2, 2), #Output is 32x64x64
            self.__downBlock(32, 128, 3, 1, 1, 4, 4) #Output is 128x16x16
        )

        self.mid_layer = nn.Sequential(
            nn.Conv2d(128,256,3,1,1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2,1),
        )

        self.up_layers = nn.Sequential(
            self.__upBlock(256, 128, 4, 2, 0), #Output is 128x16x16
            self.__upBlock(128, 32, 4, 2, 2),  #Output is 32x64x64
            self.__upBlock(32, 12, 4, 2, 1),   #Output is 12x128x128
            self.__upBlock(12, 6, 4, 2, 1),    #Output is 6x256x256
            self.__upBlock(6, 3, 4, 2, 1),     #Output is 3x512x512
        )

    def __downBlock(self, in_channels, out_channels, conv_kernel_size, conv_stride, padding, pool_kernel, pool_stride):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, conv_kernel_size, conv_stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(pool_kernel, pool_stride),
        )

    def __upBlock(self, in_channels, out_channels, conv_kernel_size, conv_stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, 
                out_channels, 
                conv_kernel_size, 
                conv_stride, 
                padding
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        # Input x has dimensions B x 3 x 512 x 512, B is batch size
        x = self.down_layers(x)
        x = self.mid_layer(x)
        x = self.up_layers(x)
        # Output has dimensions B x 3 x 512 x 512
        return x

model = Model1()
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'{total_params} Parameters in CNN')