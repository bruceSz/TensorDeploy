#!/usr/bin/env python3

import torch
from torch import nn

class SSD_VGG(nn.Module):
    __BACKBONE__ = "vgg"
    def __init__(self, num_classes, pretrained = False):
        self.num_classes = num_classes
        self.vgg = add_vgg(pretrained)
        self.extra = add_head(2014)
        self.L2Norm = L2Norm(512, 20)
        mbox = [4, 6, 6, 6, 4, 4]


def main():
    pass

if __name__ == "__main__":
    main()