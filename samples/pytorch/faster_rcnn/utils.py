#!/usr/bin/env python3

import os
import torch
import scipy

from torch.utils.tensorboard import SummaryWriter

from matplotlib import pyplot as plt
class LossHistory(object):
    def __init__(self, log_dir, model, input_shape):
        self.log_dir  = log_dir
        self.losses = []

        self.val_loss = []

        os.makedirs(log_dir)
        self.writer = SummaryWriter(log_dir)

    def append_loss(self, epoch, loss, val_loss):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.losses.append(loss)
        self.val_loss.append(val_loss)

        with open(os.path.join(self.log_dir,"epoch_loss.txt"), 'a') as f:
            f.write(str(loss))
            f.write("\n")

        with open(os.path.join(self.log_dir, "epoch_val_loss.txt"), 'a') as f:
            f.write(str(val_loss))
            f.write("\n")

        self.writer.add_scalar("loss", loss, epoch)
        self.writer.add_scalar("val_loss", val_loss, epoch)
        self.loss_plot()

    def loss_plot(self):
        iters = range(len(self.losses))
        plt.figure()
        plt.plot(iters, self.losses, 'red', linewidth=2, label = 'train loss')
        plt.plot(iters, self.val_loss, 'blue', linewidth=2, label = 'val loss')
        try:
            if len(self.losses) < 25:
                num = 15
            else:
                num = 15
            # use 3-order polynomial to fit num length window signal and filter out low
            # frequency signal.
            plt.plot(iters, scipy.signal.savgol_filter(self.losses, num, 3), 'green', linestyle='--', linewidth=2, label='smoth train loss')
            plt.plot(iters, scipy.signal.savgol_filter(self.val_loss, num, 3), 'purple', linestyle='--', linewidth=2, label='smoth val loss')


        except:
            pass

        plt.grid(True)      
        plt.xlabel('Epoch')      
        plt.ylabel('Loss')
        plt.legend(loc='upper right')

        plt.savefig(os.path.join(self.log_dir, "epoch_loss.png"))
        plt.cla()
        plt.close("all")
                





def get_classes(classes_path):
    with open(classes_path, encoding='utf-8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names, len(class_names)

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

