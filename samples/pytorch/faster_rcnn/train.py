#!/usr/bin/env python3

import sys
import math
import os
from functools import partial
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader


import numpy as np
import datetime

from torch import nn
from dataset import FasterRCNNDataSet
from utils import get_classes
from utils import LossHistory
from common.utils import weights_init

from frcnn import FasterRCNN


class FasterRCNNTrainer(nn.Module):
    def __init__(self, model,  optimizer):
        super(FasterRCNNTrainer, self).__init__()
        self.model = model
        self.optimizer = optimizer

        self.rpn_sigma = 1
        self.roi_sigma = 1

        self.anchor_target_creator = AnchorTargetCreator()
        self.proposal_target_creator = ProposalTargetCreator()

        self.loc_normalize_std = [0.1, 0.1, 0.2, 0.2]


    def _fast_rcnn_loc_loss(self, loc_pred, gt_loc, gt_label,sigma):
        # where gt_label is 1
        pred_loc = pred_loc[gt_label > 0]
        gt_loc = gt_loc[gt_label > 0]
        sigma_squared = torch.pow(sigma, 2)

        # for all with right gt_label
        reg_diff = (gt_loc - pred_loc)
        # l1 loss
        reg_diff = reg_diff.abs().float()
        reg_loss = torch.where()


def get_lr_scheduler(lr_decay_type, lr, min_lr, total_iter, warmup_iter_ratio=0.05, 
                     warmup_lr_ratio=0.1, no_aug_iter_ratio=0.05, step_num =10):
    def yolox_warm_cos_lr(lr, min_lr, total_iter, warmup_total_iter, warmup_lr_start,
                          no_aug_iter, iters):
        if iters <= warmup_total_iter:
            lr = (lr - warmup_lr_start) / pow(iters/float(warmup_total_iter),2) + warmup_total_iter*warmup_lr_start
        elif iters >= total_iter - no_aug_iter:
            lr = min_lr

        else:
            lr = min_lr + 0.5 * (lr - min_lr) * (
                1.0 + math.cos(math.pi * (iters - warmup_total_iter)/(total_iter - warmup_total_iter - no_aug_iter - no_aug_iter)))
        return lr
    
    def step_lr(lr, decay_rate, step_size, iters):
        if step_lr < 1:
            raise ValueError("step_lr must be greater than 1")
        n = iters// step_size
        out_lr = lr * decay_rate **n
        return out_lr
    
    if lr_decay_type == 'cos':
        warmup_total_iters = min(max(warmup_iter_ratio * total_iter, 1), 3)
        warmup_lr_start = max(warmup_lr_ratio * lr, 1e-6)

        no_aug_iter = min(max(no_aug_iter_ratio * total_iter, 1), 15)

        func = partial(yolox_warm_cos_lr, lr, min_lr, total_iter,warmup_total_iters,warmup_lr_start, no_aug_iter)

    else:
        decay_rate = (min_lr/ lr ) ** (1/(step_num -1))
        step_size = total_iter//step_num
        func = partial(step_lr, lr, decay_rate, step_size)
    
    return func
            #lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_iter, eta_min=min_lr)

def train():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    fp16 = False

    classes_path = "data/voc/classes_voc.txt"


    model_path = "model_checkpoints/voc_weights_resnet.pth"

    input_shape = [600, 600]

    backbone = "resnet50"

    #only valid when model_path is None
    pretrained=False

    # prior set anchor size.
    # custom according to use case.
    anchor_size = [ 8, 16, 32]

    init_epoch = 0
    freeze_epoch = 50
    freeze_batch_size = 4

    unfreeze_epoch = 100
    unfreeze_batch_size = 2

    freeze_train = True
    # for adam, set init_lr to 1e-4
    # for sgd, set init_lr to 1e-2
    init_lr = 1e-4
    min_lr = init_lr * 0.01
    
    optimizer_type = "adam"
    momentum = 0.9
    # for adam, set weight_decay = 0
    weight_decay = 0

    lr_decay_type = 'cos'

    save_period = 5

    save_dir = 'data/voc/logs'

    eval_flag = True
    eval_period = 5

    num_workers = 1

    train_anno_path = "data/voc/2007_train.txt"
    val_anno_path = "data/voc/2007_val.txt"

    class_names, num_classes = get_classes(classes_path)
    model = FasterRCNN(num_classes, anchor_scales=anchor_size, backbone=backbone, pretrained=pretrained)
    if not pretrained:
        weights_init(model)
    if model_path is not None:
        assert(os.path.exists(model_path))
        print("Loading model from {}".format(model_path))

        model_dict = model.state_dict()
        #  pre-trained weights
        # this is dict from parameter name to tensor
        pretrained_dict = torch.load(model_path, map_location=device)
        
        # for k in model_dict.keys():
        #     print("model: ",k)

        # for k in pretrained_dict.keys():
        #     print("pretrained: ",k)
        
        
        load_key, no_load_key, temp_dict = [], [],{}

        for k,v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                # k exist in model_dict and shape of v is same
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)

        model_dict.update(temp_dict)
        # reload model_state.
        model.load_state_dict(model_dict)
        #print("updated dict: ",temp_dict.keys())
        print("Loaded pre-trained weights from {}".format(model_path))
        print("No pre-trained weights to load: {}".format(no_load_key))
        
    time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
    log_dir = os.path.join(save_dir, "loss_" +time_str)
    loss_his = LossHistory(log_dir, model, input_shape=input_shape)

    if fp16:
        from torch.cuda.amp import GradScaler
        scaler = GradScaler()
    else:
        scaler = None


    model_train  = model.train()

    # for cuda, using this branch.
    if False:
        model_train = torch.nn.DataParallel(model_train)

        cudnn.benchmark  = True
        model_train = model_train.cuda()

    
    with open(train_anno_path, 'r') as f:
        train_lines = f.readlines()

    with open(val_anno_path, 'r') as f:
        val_lines = f.readlines()

    num_train = len(train_lines)
    num_val = len(val_lines)

    # show_config(
    #     classes_path = classes_path, 
    #     model_path = model_path, 
    #     input_shape = input_shape, 
    #     anchor_size = anchor_size, 
    #     init_epoch = init_epoch, 
    #     freeze_epoch = freeze_epoch,
    # )

    wanted_step = 5e4 if optimizer_type == "sgd" else 1.5e4
    total_step = num_train //  unfreeze_batch_size  * unfreeze_epoch

    if total_step < wanted_step:
        if num_train // unfreeze_batch_size == 0 :
            raise ValueError("dataset is too  small for training")
        wanted_epoch = wanted_step //(num_train//unfreeze_batch_size) + 1

        print("it is adviced that total_step should be larger than %d , when using %s optimizer"%(wanted_step, optimizer_type))
        print("In this run, total train set has %d samples, unfreeze_batch_size is %d"%(num_train, unfreeze_batch_size))
        print("All together %d steps and % d epochs"%(total_step, wanted_epoch))
        print("All train step is %d, less than %s all step, set all epoch to %d"%(total_step, wanted_step, wanted_epoch))



    if True:
        UnFreeze_flag = False
        # freeze extractor.
        if freeze_train:
            for p in model.extractor.parameters():
                p.requires_grad = False


        model.freeze_bn()

        batch_size = freeze_batch_size if freeze_train else unfreeze_batch_size

        nbs = 16

        lr_limit_max = 1e-4 if optimizer_type == "sgd" else 5e-2
        lr_limit_min = 1e-4 if optimizer_type == "sgd" else 5e-4

        Init_lr_fit = min(max(batch_size/nbs * init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit = min(max(batch_size/nbs * min_lr, lr_limit_min * 1e-2),  lr_limit_max * 1e-4)


        optimizer = {
            'adam' : optim.Adam(model.parameters(), lr=Init_lr_fit, betas=(momentum, 0.99), weight_decay=weight_decay),
            'sgd' : optim.SGD(model.parameters(), lr=Init_lr_fit, momentum=momentum, nesterov=True, weight_decay=weight_decay)
        }[optimizer_type]


        lr_scheduler = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, unfreeze_epoch)

        epoch_step = num_train//batch_size
        epoch_val_step = num_val//batch_size


        if epoch_step ==0 or epoch_val_step ==0:
            raise ValueError("dataset is too  small for training")
        
        train_dataset = FasterRCNNDataSet(train_lines, input_shape, train=True)
        val_dataset = FasterRCNNDataSet(val_lines, input_shape, train=False)


        gen = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers,
                         pin_memory=True,drop_last =True, collate_fn=frcnn_dataset_collate)
        
        gen_val = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers,
                             pin_memory=True,drop_last =True, collate_fn=frcnn_dataset_collate)
        
        train_util = FasterRCNNTrainer(model, optimizer)
        eval_callback  = EvalCallback(model, input_shape, class_names, num_classes,
                                      val_lines, log_dir, True, eval_flag=eval_flag, period=eval_period)
        

        for epoch in range(init_epoch, unfreeze_epoch):
            pass

    


if __name__ == '__main__':
    train()