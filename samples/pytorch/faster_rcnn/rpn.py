#!/usr/bin/env python3


from torch import nn

class RegionProposalNetwork(nn.Module):
    def __init__(self,
        in_channels = 512,
        mid_channels = 512,
        ratios = [0.5, 1, 2] ,
        anchor_scales = [8, 16, 32],
        feat_stride=16,
        mode="training"):
    
        super(RegionProposalNetwork, self).__init__()
        self.anchor_base = genera_anchor_base(anchor_scales, ratios)
        

    def generate_anchor_base(self, anchor_scales, ratios):