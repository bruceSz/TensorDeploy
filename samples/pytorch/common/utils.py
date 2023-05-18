#!/usr/bin/env python3

import time
import torch

class MyTime(object):
    def __init__(self) -> None:
        pass

    def __enter__(self):
        self.start = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        curr = time.time()
        print("elapsed time: ", curr - self.start)
        self.start = None


def weights_init(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        class_name = m.__class__.__name__
        #conv layer
        # gain can be computed by torch.nn.init.calculate_gain
        #1. in xavier uniform initializer, weight satisify (−a,a) uniform distribution
        #  where a = gain * sqrt(6/fan_in+fan_out)，
        

        if hasattr(m, 'weight') and class_name.find("Conv") != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, mean=0.0, gain=init_gain)
            elif init_type == 'xavier':
                #2. in xavier normal initializer, weight satisify normal,
                #   where mean=0, std = sqrt(2/fan_in+fan_out)
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                # 0-mean normal, std = sqrt(2/(1+a^2)*fan_in)
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                # make the weight tensor orthogonal.
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif class_name.find("BatchNorm2d")!= -1:
            torch.nn.init.normal_(m.weight.data, mean=1, std=0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)
    
    print("init network with %s" % init_type)
    # apply the initialization to each layer in the network
    net.apply(init_func)