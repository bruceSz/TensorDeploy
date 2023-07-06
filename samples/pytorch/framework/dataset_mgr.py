#!/usr/bin/env python3

import sys
import os
import time

sys.path.append(os.path.dirname(os.path.abspath(sys.path[0])))

from faster_rcnn.utils import get_classes
from common.utils import singleton

class VOC_Dataset(object):
    def __init__(self) -> None:
        self.classes_path = "data/voc/classes_voc.txt"
        self.train_path = "data/voc/2007_train.txt"
        self.val_path = "data/voc/2007_val.txt"
        #self.class_names, self.num_classes = self._get_class(self.classes_path)
        self.class_names, self.num_classes = get_classes(self.classes_path)

    

@singleton
class DataSetMgr(object):
    #VOC_DATASET_DIR = ""
    def __init__(self) -> None:
        self.date =  time.time()

    def get_dataset(self, name):
        if name == "voc":
            return VOC_Dataset()
        else:
            raise NotImplementedError("Unknown dataset: {}".format(name))
    
def main():
    dsm = DataSetMgr()
    print("dsm ts: ", dsm.date)
    voc_dataset = dsm.get_dataset("voc")
    print(voc_dataset.class_names)
    print(voc_dataset.num_classes)

if __name__ == "__main__":
    main()