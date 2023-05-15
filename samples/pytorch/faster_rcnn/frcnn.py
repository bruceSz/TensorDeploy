#!/usr/bin/env python3

import torch.nn as nn
import resnet50
from rpn import RegionProposalNetwork

class FasterRCNN(nn.Module):
    def __init__(self, n_classes,
                mode = 'train',
                feat_stride = 16,
                anchor_scales = [8, 16, 32],
                ratios = [0.5, 1, 2],
                backbone = 'resnet50',
                pretrained=False):
        super(FasterRCNN, self).__init__()
        self.feat_stride = feat_stride

        # only one backbone: resnet50

        self.extractor, classifier = resnet50.resnet50(pretrained)

        self.rpn = RegionProposalNetwork(1024, 512,
        anchor_scales = anchor_scales,
        ratios = ratios,
        feat_stride = feat_stride)

        self.head = Resnet50ROIHead(
            n_classes = n_classes + 1,
            roi_size = 14, 
            spatial_scale = 1,
            classifier = classifier
        )

