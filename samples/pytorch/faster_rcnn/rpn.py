#!/usr/bin/env python3


import torch
from typing import Any
from torch import nn
from torchvision import nms
from torch.nn import functional as F
import numpy as np
from anchors  import generate_anchor_base, _compute_all_shifted_anchors
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

        min_size = self.min_size * scale
        # return of torch.where is a tuple with len == 1
        keep = torch.where(((roi[:,2] - roi[:,0]) >= min_size) &
                           ((roi[:,3] - roi[:,1]) >= min_size))[0]
        
        # only keep bbox above size threshold
        # only keep score corresponding to above boxes.
        roi = roi[keep,:]
        score = score[keep]

        order = torch.argsort(score, descending=True)
        if pre_nms_top_n > 0:
            order = order[:pre_nms_top_n]

        roi = roi[order,:]
        score = score[order]

        keep = nms(roi, score, self.nms_iou)

        # if iou is too big(strict)
        # duplicate some of keep by sample from keep list randomly.
        if len(keep) < post_nms_top_n:
            index_extra = np.random.choice(range(len(keep)), size=(post_nms_top_n - len(keep)), inplace=True)
            keep = np.cat([keep, keep[index_extra]])
        
        keep = keep[:post_nms_top_n]
        roi = roi[keep]
        return roi



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

        self.normal_init(self.conv1, 0, 0.01)
        self.normal_init(self.score, 0, 0.01)
        self.normal_init(self.loc, 0, 0.01)

    def forward(self, x, img_size, scale=1.):
        n, _, h, w = x.shape

        x = F.relu(self.conv1(x))

        rpn_locs = self.loc(x)
        #transpose/permute channel dimension to last dimension which has length of 4.
        rpn_locs = rpn_locs.permute(0,2,3,1).contiguous().view(n, -1, 4)

        rpn_scores = self.score(x)
        # same as rpn_locs, permute/transpose channel dimension to last dimension which has length of 2.
        rpn_scores = rpn_scores.permute(0,2,3,1).contiguous().view(n, -1, 2)

        rpn_softmax_scores = F.softmax(rpn_scores, dim=-1)
        rpn_fg_scores = rpn_softmax_scores[:, :, 1]
        rpn_fg_scores = rpn_fg_scores.view(n, -1, 1)

        anchor = _compute_all_shifted_anchors(np.array(self.anchor_base), self.feat_stride, h, w)
        rois = list()
        roi_indices = list()
        for i in range(n):
            roi = self.proposal_layer(rpn_locs[i], rpn_fg_scores[i], anchor,img_size,scale=scale)
            batch_index = i * torch.ones((len(roi),))
            rois.append(roi.unqueeze(0))
            roi_indices.append(batch_index.squeeze(0))

        rois = torch.cat(rois,dim=0).type_as(x)
        roi_indices = torch.cat(roi_indices,dim=0).type_as(x)

        anchor = torch.from_numpy(anchor).unsqueeze(0).float().to(x.device)

        return rpn_locs, rpn_scores, rois, roi_indices, anchor


        

    def normal_init(self, m, mean, stddev, truncated=False):
        if truncated:
            m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)
        else:
            m.weight.data.normal_(mean, stddev)
            m.bias.data.zero_()