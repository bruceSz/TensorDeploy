#!/usr/bin/env python

from typing import Any
import torch

from torch import nn

from dataset import get_loaders
from config import Config
from backbones.blocks import CBRBlock

# differences between VGG16 and VGG19: 16 hidden layers vs 19 hidden layers, 
# prediction heads are same: 3 layer fc.

class VGG_HEAD(nn.Module):
    def __init__(self, in_channel ,n_classes, *args, **kwargs) -> None:
        super(VGG_HEAD, self).__init__(*args, **kwargs)
        self.inc = in_channel
        self.fc = nn.Sequential(
            nn.Linear(in_channel , in_channel),
            nn.ReLU(inplace=True),
            # default it 0.5
            nn.Dropout(0.5),

            nn.Linear(in_channel, in_channel),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(in_channel, n_classes)
        )

    def forward(self, x):
        x = x.view(-1, self.inc)
        return self.fc(x)
    

class VGG16_FT(nn.Module):
    def __init__(self, *args: Any, **kwds: Any) -> Any:
        super(VGG16_FT, self).__init__(*args, **kwds)
        
        self.layer1 = nn.Sequential(
            # in channes: 3, out channels: 64,  kernel size: 3,stride: 1, padding: 1
            CBRBlock(3, 64, 3, 1, 1),
            # in channes: 64, out channels: 64,kernel size: 3
            # stride: 1
            # padding: 1
            CBRBlock(64, 64, 3, 1, 1),
            nn.MaxPool2d(2, 2, ceil_mode=True)
        )

        self.layer2 = nn.Sequential(
            # in channes: 64, out channels: 128, kernel size: 3, stride: 1, padding: 1
            CBRBlock(64, 128, 3, 1, 1),
            # in channes: 128, out channels: 128, kernel size: 3, stride: 1, padding: 1
            CBRBlock(128, 128, 3, 1, 1),
            nn.MaxPool2d(2, 2, ceil_mode=True)
        )

        self.layer3 = nn.Sequential(
            # in channes: 128, out channels: 256, kernel size: 3, stride: 1, padding: 1
            CBRBlock(128, 256, 3, 1, 1),
            CBRBlock(256, 256, 3, 1, 1),
            CBRBlock(256, 256, 3, 1, 1),
            nn.MaxPool2d(2, 2, ceil_mode=True)
        )

        self.layer4 = nn.Sequential(
            # in channes: 256, out channels: 512, kernel size: 3, stride: 1, padding: 1
            CBRBlock(256, 512, 3, 1, 1),
            CBRBlock(512, 512, 3, 1, 1),
            CBRBlock(512, 512, 3, 1, 1),
            nn.MaxPool2d(2, 2, ceil_mode=True)
        )
        self.layer5 = nn.Sequential(
            # in channes: 512, out channels: 512, kernel size: 3, stride: 1, padding: 1
            CBRBlock(512, 512, 3, 1, 1),
            CBRBlock(512, 512, 3, 1, 1),
            CBRBlock(512, 512, 3, 1, 1),
            nn.MaxPool2d(2, 2, ceil_mode=True)
        )

        self.conv = nn.Sequential(
            self.layer1, 
            self.layer2,
            self.layer3,
            self.layer4,
            self.layer5
        )
    def forward(self, x):
        return self.conv(x)

        

class VGG16_CF(nn.Module):
    def __init__(self,  *args, **kwargs) -> None:
        super(VGG16_CF, self).__init__(*args, **kwargs)
        self.ft_layer = VGG16_FT()
        self.head = VGG_HEAD(512, 10)

    def forward(self, x):
        x = self.ft_layer(x)
        print("ft shape:",x.shape)
        x = self.head(x)
        print("head shape:",x.shape)
        return x
    

if __name__ == "__main__":
    model = VGG16_CF()
    print(model)
    conf = Config()
    _, val_loader = get_loaders(conf)
    
    for inputs, labels in val_loader:
        print(inputs.shape)
        #output = model(inputs)
        print(labels.shape)
        break
    input = torch.randn(1, 3, 32, 32)
    print("input is :", input.shape)
    output = model(input)
    print(output.shape)