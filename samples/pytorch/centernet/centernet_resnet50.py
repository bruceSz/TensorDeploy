#!/usr/bin/env python3

import sys
import os
import math
from torch import nn
sys.path.append(os.path.dirname(os.path.abspath(sys.path[0])))

from backbones.resnet50 import resnet50
from resnet50_adaptor import resnet50_Decoder, resnet50_Head

class CenterNet_Resnet50(nn.Module):
    __BACKBONE__ = "resnet50"
    def __init__(self, n_class = 20, pretrained = False) -> None:
        super(CenterNet_Resnet50, self).__init__()
        self.pretrained = pretrained
        # 512,512,3 -> 16 * 16 * 2048
        self.backbone = resnet50(pretrained= pretrained)

        # 16, 16, 2048 -> 128 , 128, 64
        self.decoder = resnet50_Decoder(2048)

        self.head = resnet50_Head(channel=64, num_classes= n_class)

        self._init_weights()

    def _init_weights(self):
        if not self.pretrained:
            for m in self.modules:
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    # init with MSRC initializer/kaiming
                    m.weight.data.normal_(0, math.sqrt(2./n))
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
        #TODO: last conv2d?
        self.head.cls_head[-1].weight.data.fill_(0)
        self.head.cls_head[-1].bias.data.fill_(0)



        
    def forward(self, x):
        feat = self.backbone(x)
        return self.head(self.decoder(feat))
    


class CenterNet_HourglassNet(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        raise NotImplementedError("HourglassNet based centernet is not implemented yet.")