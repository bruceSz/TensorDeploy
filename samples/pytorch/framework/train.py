#!/usr/bin/env python3
import torch
import argparse

from framework import model_mgr

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--faster_rcnn', help='train faster rcnn')
    parser.add_argument('--centernet', help='train centernet')
    parser.add_argument('--dataset', help='dataset to be used')
    parser.add_argument('--epochs', help='number of epochs')
    parser.add_argument('--batch_size', help='batch size')
    parser.add_argument('--multi_gpu', help='train with multi-gpu')
    parser.add_argument('--fp16', help='use fp16 training')
    parser.add_argument('--pretrained', help='pretrained model')
    return parser


class TrainConfig(object):
    def __init__(self, args) -> None:
        self.init_epoch = 0
        self.freeze_epoch = 50
        self.freeze_batch_size = 16
        self.unfreeze_epoch = 100
        self.unfreeze_batch_size = 8
        self.freeze_train = True
        self.init_lr = 5e-4
        self.min_lr = self.init_lr * 1e-2
        self.opt_type = "adam"
        self.mementum =  0.9
        self.weight_decay =  0
        self.lr_decay_type = 'cos'
        self.save_period = 5
        self.save_dir = "logs"
        self.eval_flag = True
        self.eval_period = 5
        self.num_workers = 4
        self.distributed = False
        self.backbone_name = "restnet50"

        if self.distributed:
            #TODO
            raise NotImplementedError("Distributed training is not implemented yet")
        else:
            self.device = torch.device('cuda' if torch.cuda.device_count() > 0 else 'cpu')
            self.local_rank = 0
        self.n_gpus = torch.cuda.device_count()

def train():
    parser = create_parser()
    args = parser.parse_args()
    print(args)

   
    tc = TrainConfig(args)
   

    model = model_mgr.ModelManager(tc)

    model.init()

   

    














if __name__ == '__main__':
    train()