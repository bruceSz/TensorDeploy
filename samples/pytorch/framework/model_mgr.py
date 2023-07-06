#!/usr/bin/env python3

import torch.distributed  as dist

import dataset_mgr

class BackboneMgr(object):
    def __init__(self, tc) -> None:
        #self.name = tc.backbone_name
        #self.dataset_mgr = dataset_mgr.DataSetMgr()
        #self.num_classes = self.dataset_mgr.get_dataset(tc.dataset_name).num_classes
        #self.tc = tc
        self.backbones = {}

    def init_with_download(cls):
        #TODO
        raise NotImplementedError("init backbone via downloading is not implemented yet")
        #download_weights(backbone)

    def get_model(self, model_cfg):
        #if name == "resnet50":
        return self.backbones[model_cfg.name](model_cfg)
            #return CenterNet_Resnet50(self.num_classes, pretrained=self.tc.pretrained)
            
    # def init_locally(cls, name):
    #     if name == "resnet50":
    #         self.model = CenterNet_Resnet50(self.num_classes, pretrained=self.tc.pretrained)

    def RegisterBackbone(self, backbone):
        self.backbones[backbone.__BACKBONE__] = backbone


class ModelManager(object):
    def __init__(self, tc) -> None:
        self.pretrained = tc.pretrained
        self.distributed = tc.distributed
        self.tc = tc
        self.backbone_mgr = BackboneMgr(tc)

    def init(self):
        if self.pretrained:
            if self.distributed:
                if self.tc.local_rank == 0:
                    #print("=> using pre-trained model '{}'".format(self.pretrained))
                    self.backbone_mgr.init_with_download()
                dist.barrier()
            else:
                self.backbone_mgr.init_with_download()

