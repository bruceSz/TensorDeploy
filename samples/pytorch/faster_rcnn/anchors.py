#!/usr/bin/env python3


def generate_anchor_base(base_size=16, ratios = [0.5, 1, 2], anchor_scales = [8, 16, 32]):
    # each anchor: four number, top-left(h,w) bottom-right(h,w)
    anchor_base = np.zeros((len(ratios), len(anchor_scales), 4))
    for i in range(len(ratios)):
        for j in range(len(anchor_scales)):
            h = base_size * anchor_scales[j] * np.sqrt(ratios[i])
            w = base_size * anchor_scales[j] * np.sqrt(1./ratios[i])

            index = i * len(anchor_scales) + j
            anchor_base[i, j, 0] =  - h/2.
            anchor_base[i, j, 1] =  - w/2.
            anchor_base[i, j, 2] =  h/2.
            anchor_base[i, j, 3] =  w/2.
    return anchor_base



def _compute_all_shifted_anchors(anchor_base, feat_stride, height, width):
    """
        echo feature_map point map to a feat_stride * feat_stride grid in original image space,

        anchor_base: [9, 4], A ==9
        feat_stride: 16
        feature_map shape: height * width

    """

    shift_x = np.arange(0,width* feat_stride,feat_stride)
    shift_y = np.arange(0,height* feat_stride,feat_stride)
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    # shift_x, shift_y:  height * width shape mat.
    # shift is all combination of feature_map points in original image
    # shape of shift: K * 4
    shift = np.stack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel()), axis=1)

    # A: anchor number 
    A = anchor_base.shape[0]
    # K: number of grid in original image
    K = shift.shape[0]

    anchor = anchor_base.reshape((1,A,4))  + shift.reshape((K,1,4))
    # each shift will be added to anchor of anchor_base (short form of double loop)
    # shape of anchor: K * A * 4

    anchor = anchor.reshape((K*A,4)).astype(np.float32)
    
    return anchor

    