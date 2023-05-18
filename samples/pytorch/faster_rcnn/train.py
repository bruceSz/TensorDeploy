#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch


import numpy as np
import datetime

from utils import get_classes
from common.utils import weights_init

from frcnn import FasterRCNN



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
        pretrained_dict = torch.load(model_path, map_location=device)
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
        print("Loaded pre-trained weights from {}".format(model_path))
        print("No pre-trained weights to load: {}".format(no_load_key))
        
    time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
    log_dir = os.path.join(save_dir, "loss_" +time_str)
    loss_his = LossHistory(log_dir, model, input_shape=input_shape)








if __name__ == '__main__':
    train()