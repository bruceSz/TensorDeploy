#!/usr/bin/env python3


import torch
from typing import Any
from torch import nn
from anchors  import generate_anchor_base
from utils import loc2box

class ProposalCreator(object):
    def __init__(self,
                 mode, 
                 nms_iou = 0.7,
                 n_train_pre_nms =12000 ,
                 n_train_post_nms = 600,
                 n_test_pre_nms = 3000,
                 n_test_post_nms =300,
                 min_size = 16) -> None:
        self.mode = mode
        self.nms_iou = nms_iou
        
        self.n_train_pre_nms = n_train_pre_nms
        self.n_train_post_nms = n_train_post_nms

        self.n_test_pre_nms = n_test_pre_nms
        self.n_test_post_nms = n_test_post_nms

        self.min_size = min_size


    def __call__(self, loc, score, anchor,  img_size, scale=1.) -> Any:
        if self.mode == "training":
            pre_nms_top_n = self.n_train_pre_nms
            post_nms_top_n = self.n_train_post_nms
        elif self.mode == "test":
            pre_nms_top_n = self.n_test_pre_nms
            post_nms_top_n = self.n_test_post_nms
        else:
            raise ValueError("Unknown mode: {}".format(self.mode))

        anchor = torch.from_numpy(anchor).type_as(loc)
        roi = loc2box(anchor, loc)
        # clamp x
        roi[:, [0,2]] = torch.clamp(roi[:[0,2]], min=0, max=img_size[1])
        # clamp y
        roi[:, [1,3]] = torch.clamp(roi[:[1,3]], min=0, max=img_size[0])

        #TODO.



class RegionProposalNetwork(nn.Module):
    def __init__(self,
        in_channels = 512,
        mid_channels = 512,
        ratios = [0.5, 1, 2] ,
        anchor_scales = [8, 16, 32],
        feat_stride=16,
        mode="training"):
    
        super(RegionProposalNetwork, self).__init__()
        self.anchor_base = generate_anchor_base(anchor_scales, ratios)
        n_anchors = self.anchor_base.shape[0]

        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1)
        self.score = nn.Conv2d(mid_channels, n_anchors*2, kernel_size=1, stride=1, padding=0)

        self.loc = nn.Conv2d(mid_channels, n_anchors*4, kernel_size=1, stride=1, padding=0)

        self.feat_stride = feat_stride

        self.proposal_layer = ProposalCreator(mode)
        

    def generate_anchor_base(self, anchor_scales, ratios):
        