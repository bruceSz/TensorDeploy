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

    print("loc shape under loc2box: ", loc.shape)
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



import numpy as np



def get_new_img_size(height, width, img_min_side=600):
    if width <= height:
        f = float(img_min_side) / width
        resized_height = int(f * height)
        resized_width = int(img_min_side)
    else:
        f = float(img_min_side) / height
        resized_width = int(f * width)
        resized_height = int(img_min_side)
    return resized_height, resized_width

class DecodeBox(object):
    def __init__(self, std, num_classes):
        self.std = std
        self.num_classes = num_classes

    def frcnn_correct_boxes(self, box_xy, box_wh, input_shape, image_shape):

        # here we put y axis in front of x axis
        # in that it is more convenient to multi pred-box's width-height
        box_yx = box_xy[..., ::-1]
        box_hw = box_wh[..., ::-1]

        input_shape  = np.array(input_shape)
        image_shape = np.array(image_shape)

        box_mins = box_yx - (box_hw / 2.)
        box_maxes = box_yx + (box_hw / 2.)

        boxes = np.concatenate([box_mins[..., 0:1], 
                    box_mins[..., 1:2], 
                    box_maxes[..., 0:1], box_maxes[..., 1:2]], axis=-1)
        boxes *= np.concatenate([image_shape, image_shape], axis=-1)

        return boxes

    def forward(self, roi_cls_locs, roi_scores, rois, image_shape,
                input_shape, nms_iou = 0.3, confidence = 0.5):
        results = []

        # return outer-most dimension.
        bs = len(roi_cls_locs)

        # batch_size, num_rois, 4
        rois = rois.view(bs, -1, 4)

        for i in range(bs):
            # reshape regression parameters
            roi_cls_loc = roi_cls_locs[i] * self.std

            # [num_rois, num_classes, 4]
            roi_cls_loc = roi_cls_loc.view([-1, self.num_classes, 4])


            roi = rois[i].view((-1, 1, 4)).expand_as(roi_cls_loc)
            cls_bbox = loc2bbox(roi.contiguous().view((-1,4)), roi_cls_loc.contiguous().view(-1,4))
            cls_bbox = cls_bbox.view([-1, (self.num_class), 4])

            # normalize pred-bbox
            # input_shape-> height-width?
            cls_bbox[..., [0,2]] = (cls_bbox[...,[0,2]])/input_shape[1]
            cls_bbox[...,[1,3]] = (cls_bbox[...,[1,3]])/input_shape[0]


            roi_score = roi_scores[i]
            prob = F.softmax(roi_score, dim=-1)

            results.append([])
            for c in range(1, self.num_classes):
                # threshold check

                c_confs = prob[:,c]
                c_confs_m = c_confs > confidence

                if len(c_confs[c_confs_m]) > 0:
                    boxes_to_process = cls_bbox[c_confs_m, c]
                    confs_to_process = c_confs[c_confs_m]

                    keep = nms(
                        bboxes_to_process,
                        confs_to_process,
                        nms_iou
                    )


                    # get good box from results of nms
                    good_boxes = boxes_to_process[keep]
                    # why we need None?
                    confs = confs_to_process[keep][:,None]
                    if confs.is_cuda:
                        labels = (c-1)  * torch.ones((len(keep), 1)).cuda()  
                    else:
                        labels = ( c-1) * torch.ones((len(keep),1))

                    c_pred = torch.cat((good_boxes, confs, labels), dim=1).cpu().numpy()
                    results[-1].append(c_pred)

            if len(results[-1]) > 0:
                results[-1] = np.array(results[-1])
                box_xy, box_wh = (results[-1][:,0:2] + results[-1][:,2:4])/2,\
                                    results[-1][:,2:4] - results[-1][:,0:2]
                results[-1][:,:4] = self.frcnn_correct_boxes(box_xy, box_wh, input_shape, image_shape)
        return results


def init_model( model_path, model, device):
    import os
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

class EvalCallback(object):
    def __init__(self, net, input_shape, class_names, num_classes, val_lines, log_dir, cuda\
                 ,map_out="./tmp_out", max_boxes = 100, confi=0.05, nms_iou=0.5\
                 , letterbox=True, MIN_OVERLAP=0.5, eval_flag=True, period=1 ):
        super(EvalCallback, self).__init__()

        self.net = net
        self.input_shape = input_shape
        self.class_names = class_names
        self.num_classes = num_classes
        self.val_lines = val_lines
        self.log_dir = log_dir
        self.cuda = cuda
        self.map_out = map_out
        self.max_boxes = max_boxes
        self.confidence = confi
        self.nms_iou = nms_iou
        self.letterbox_img = letterbox
        self.MIN_OVERLAP = MIN_OVERLAP
        self.eval_flag = eval_flag
        self.period = period


        self.std = torch.Tensor([0.1, 0.1, 0.2, 0.2]).repeat(self.num_classes + 1)
        if self.cuda:
            self.std = self.std.cuda()
        self.bbox_util = DecodeBox(self.std, self.num_classes)

        self.maps = [0]
        self.epoches = [0]
        
        if self.eval_flag:
            with open(os.path.join(self.log_dir, "epoch_loss.txt"), 'a') as f:
                f.write(str(0))
                f.write("\n")

    def get_map_txt(self, img_id, img, class_names, map_out):
        f = open(os.path.join(map_out, "det-results/"+img_id+".txt"), 'w')

        # compute image height and width
        img_shape = np.array(np.shape(img)[0:2])
        # change to (600, width), or (height, 600)
        input_shape = get_new_img_size(img_shape[0],  img_shape[1])

        # cvt to rgb
        img = cvtColor(img)

        # resize  to width == 600
        img_data = resize_img(img, [input_shape[1], input_shape[0]])

        # add batch dimension
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(img_data, dtype='float32')) , (2,0,1)),0)


        # ready to predict.
        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            
            roi_cls_locs, roi_scores, rois, _ = self.net(images)
            # decode predictions
            results = self.bbox_util.forward(roi_cls_locs, roi_scores,
                                            image_shape,input_shape,
                                            nms_iou=self.nms_iou,confidence=self.confidence)
            if len(results[0]) <= 0:
                return
            top_label = np.array(results[0][:,5], dtype='int32')
            top_conf = results[0][:,4]
            top_boxes = results[0][:,:4]

        top_100 = np.argsort(top_conf)[::-1][:self.max_boses]
        top_boxes = top_boxes[top_100]
        top_confi =   top_conf[top_100]
        top_label = top_label[top_100]

        for i, c in list(enumerate(top_label)):
            pred_class = self.class_names[int(c)]
            box = top_boxes[i]
            score = str(top_confi[i])

            top, left, bottom, right = box
            if pred_class not in class_names:
                continue
            f.write("%s %s %s %s %s %s\n" % (pred_class,score[:6], str(int(left)), str(int(top)),str(int(right)), str(int(bottom))))

        f.close()

        return

    def on_epoch_end(self, epoch):
        if epoch % self.period ==0 and self.eval_flag:
            if not os.path.exists(self.map_out):
                os.makedirs(self.map_out)
            if not os.path.exists(os.path.join(self.map_out, "ground-truth")):
                os.makedirs(os.path.join(self.map_out, "ground-truth"))
            if not os.path.exists(os.path.join(self.map_out, "det-results")):
                os.makedirs(os.path.join(self.map_out, "det-results"))

            print("gget map.")

            for ann_line in tqdm(self.val_lines):
                line = ann_line.strip().split()
                img_id = os.path.basename(line[0])

                img = Image.open(line[0])

                gt_bboxes = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])

                self.get_map_txt(img_id, img, self.class_names, self.map_out)

                with open(os.path.join(self.map_out, "ground-truth/"+img_id+".txt"), 'w') as f:
                    for box in gt_bboxes:
                        left, top, right, bottom, obj = box
                        obj_name = self.class_names[obj]
                        f.write("%s %s %s %s %s\n" % (obj_name, left, top, right, bottom))

            print("calculate map.")
        
            try:
                temp_map = get_coco_map(class_names = self.class_names, path = self.map_out)[1]
            except:
                temp_map = get_map(self.MINOVERLAP, False, path=self.map_out)

            self.maps.append(temp_map)
            self.epoches.append(epoch)

            with open(os.path.join(self.log_dir, "epoch_map.txt"), 'a') as f:
                f.write(str(temp_map))
                f.write("\n")
            plt.grid(True)
            plt.xlabel("Epoch")
            plt.ylabel("Map %s"%str(self.MINOVERLAP))
            plt.title("A Map Curve")
            plt.legend(loc='upper right')
            
            plt.savefig(os.path.join(self.log_dir, "epoch_map.png"))
            plt.cla()
            plt.close("all")

            print("save map.")
            shutil.rmtree(self.map_out)



