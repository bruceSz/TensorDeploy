#!/usr/bin/env python3

import torch



def normal_init( m, mean, stddev, truncated=False):
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()

def loc2box(anchor, loc):
    """
        anchor: (tlx, tly, brx, bry)
        loc: (dx, dy, dw, dh) , prediction result.
    """
    src_width = torch.unsqueeze(anchor[:,2] - anchor[:,0], -1)
    src_height = torch.unsqueeze(anchor[:,3] - anchor[:,1], -1)
    src_c_x = torch.unsqueeze(anchor[:,0], -1) + 0.5 * src_width
    src_c_y = torch.unsqueeze(anchor[:,1], -1) + 0.5 * src_height

    dx = loc[:, 0::4]
    dy = loc[:, 1::4]
    dw = loc[:, 2::4]
    dh = loc[:, 3::4]

    ctr_x = dx * src_width + src_c_x

    ctr_y = dy * src_height + src_c_y

    w = torch.exp(dw) * src_width
    h = torch.exp(dh) * src_height

    dst_bbox = torch.zeros_like(loc)
    dst_bbox[:, 0::4] = ctr_x - 0.5 * w
    dst_bbox[:, 1::4] = ctr_y - 0.5 * h
    dst_bbox[:, 2::4] = ctr_x + 0.5 * w
    dst_bbox[:, 3::4] = ctr_y + 0.5 * h

    return dst_bbox

