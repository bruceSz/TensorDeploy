#!/usr/bin/env python3

import torch
from torch import nn

def add_vgg(pretrained = False):
    layers = []
    in_channels = 3
    for v in base:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == "C":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]

class SSD_VGG(nn.Module):
    __BACKBONE__ = "vgg"
    def __init__(self, num_classes, pretrained = False):
        self.num_classes = num_classes
        self.vgg = add_vgg(pretrained)
        self.extra = add_head(2014)
        self.L2Norm = L2Norm(512, 20)
        mbox = [4, 6, 6, 6, 4, 4]

        loc_layers = []
        conf_layers = []
        backbone_source = [21, -2]

        for k,v in enumerate(backbone_source):
            loc_layers += [nn.Conv2d(self.vgg[v].out_channels,mbox[k] * 4, kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(self.vgg[v].out_channels,mbox[k] * num_classes, kernel_size=3, padding=1)]



def main():
    pass

if __name__ == "__main__":
    main()