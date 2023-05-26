#!/usr/bin/env python3
import cv2
from torch.utils.data.dataset import Dataset
from PIL import Image
import numpy as np
import torch

def cvtColorGaurd(image):
    # hwc, where  c == 3
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image
    else:
        image = image.convert('RGB')
        return image

class FasterRCNNDataSet(Dataset):
    def __init__(self, annotation_lines, input_shape=[600, 600], train=True) -> None:
        #
        self.annotation_lines = annotation_lines
        self.input_shape = input_shape
        self.train = train
        self.len = len(self.annotation_lines)


    def __len__(self):
        return self.len
    

    def __getitem__(self, index):
        index = index % self.len


        image, y  = self.get_random_image(self.annotation_lines[index],self.input_shape[0:2], ramdom=self.train)

    def shuffle_box(self, box, dx, dy, nw, nh, iw, ih, w, h, flip=False):
        assert(len(box) > 0)
        np.random.shuffle(box)
        box[:,[0,2]] = box[:,[0,2]]*nw/iw + dx
        box[:,[1,3]] = box[:,[1,3]]*nh/ih + dy
        if flip:
            box[:,[0,2]] = w - box[:,[2,0]]
        box[:,0:2][box[:,0:2]<0] = 0
        box[:,2][box[:,2] > w] = w
        box[:,3][box[:,3]> h] =h
        box_w = box[:,2] - box[:,0]
        box_h = box[:,3] - box[:,1]
        box = box[np.logical_and(box_w>1, box_h>1)]
        return box

    def get_random_image(self, annotation_line, input_shape, random=True, jitter=0.3,
                         hue=0.1,sat=0.7, val=0.4):
        # empty space split.
        line = annotation_line.split()
        image = Image.open(line[0])
        image = cvtColorGaurd(image)
        iw, ih = image.size
        # target h, w
        h,w  = input_shape
        box = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])

        if not random:
            scale = min(w/iw, h/ih)
            nw = int(iw * scale)
            nh = int(ih * scale)
            # w-nw == left padding + right padding
            # therefore dx  = (w-nw)//2
            dx = (w - nw) // 2
            dy = (h - nh) // 2

            #bicubit interpolation
            image = image.resize((nw, nh), Image.BICUBIC)
            # gray: (128,128,128)
            new_img=  Image.new('RGB',(w,h),(128,128,128))

            new_img.paste(image,(dx,dy))

            image_data = np.array(new_img, np.float32)

            if len(box) >0:
                self.shuffle_box(box, dx, dy, nw, nh, iw, ih, w, h, flip=False)
                # np.random.shuffle(box)
                # #x multiply by scale
                # box[:,[0,2]] = box[:,[0,2]]*nw/iw + dx
                # box[:,[1,3]] = box[:,[1,3]]*nh/ih + dy

                # # left,up x,y <0 , set to 0
                # box[:,0:2][box[:,0:2]<0] = 0
                # box[:,2][box[:,2]> w] = w
                # box[:,3][box[:,3]> h] = h

                # box_w= box[:,2] - box[:,0]
                # box_h= box[:,3] - box[:,1]

                # box = box[np.logical_and(box_w>1, box_h>1)]

            # return iamge_data (resized to input_shape) and normalized box
            return image_data, box
        # in random mode, we resize and change length or width of image.
        new_ar = iw/ih * self.rand(1-jitter, 1+jitter)/self.rand(1-jitter, 1+jitter)
        scale = self.rand(0.25, 2)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw/new_ar)


        image = image.resize((nw, nh), Image.BICUBIC)
        # randomly chose dy, dy
        dx = int(self.rand(0,w-nw))
        dy = int(self.rand(0, h-nh))

        new_image = Image.new('RGB', (w,h), (128,128,128))

        new_image.paste(image,(dx,dy))
        image = new_image

        # 
        flip = self.rand() < 0.5
        if flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        img_data = np.array(image, np.uint8)

        r = np.random.uniform(-1,1,3) * [hue, sat, val] + 1

        hue, sat, val = cv2.split(cv2.cvtColor(img_data,cv2.COLOR_BGR2HSV))

        dtype = image_data.dtype

        x = np.arange(0,256, dtype=dtype)
        lut_hue = ((x * r[0]) %180).astype(dtype)
        lut_sat = np.clip(x* r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x*r[2], 0,255).astype(dtype)

        image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2BGR)

        if len(box) >0:
            self.shuffle_box(dx,dy,nw,nh, iw, ih, w, h, flip=flip)

        return image_data, box


                               

        
    def frcnn_dataset_collate(batch):
        images = []
        bboxes = []
        labels = []

        for img, box, label in batch:
            images.append(img)
            bboxes.append(box)
            labels.append(label)
        images = torch.from_numpy(np.array(images))

        return images, bboxes, labels