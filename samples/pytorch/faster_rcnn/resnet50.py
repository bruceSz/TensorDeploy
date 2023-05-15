#!/usr/bin/env python3

import math
import torch.nn as nn

class BottleNeck(nn.Module):
    expansion = 4
    def __init__(self, in_planes, planes, stride=1, downsample=False):
        super(BottleNeck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1,stride=stride , bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride


    def forward(self, x):
        res = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """
        refer: https://blog.csdn.net/zjc910997316/article/details/102912175
    """
    def __init__(self, block, layers, num_classes=100):
        self.inplanes = 64
        super(ResNet, self).__init__()
        
        
        # for input with 600 * 600 * 3 -> 300 * 300 * 64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        # 300* 300 * 64 -> 150 * 150 * 64
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)


        # 150* 150 * 64 -> 150 * 150 * 256(64 * 4)
        self.layer1 = self._make_layer(block, 64, layers[0])
        # 150* 150 * 256 -> 75 * 75 * 512 (128 * 4)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        # 75* 75 * 512 -> 38 * 38 * 1024 (256 * 4)
        # this is a shared feature map layer.
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        # 38 * 38 * 1024 -> 19 * 19 * 2048
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool =  nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)


        for m in self.modules():
            # init with MSRC initializer/kaiming
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2./n))

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x



    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)
            

GLOBAL_RESNET_PATH = "/disk2/TensorD/samples/pytorch/model_checkpoints/voc_weights_resnet.pth"

def resnet50(pretrained=False):
    model = ResNet(BottleNeck, [3, 4, 6, 3])
    if pretrained:
        model.load_state_dict(torch.load(GLOBAL_RESNET_PATH))
        #TODO read from local path
        #model.load_state_dict((model_urls['resnet50']))
        pass
    
    features = list([model.conv1, model.bn1, model.relu, model.maxpool, model.layer1, model.layer2, model.layer3])
    classifier = list([model.layer4, model.layer4])
    features = nn.Sequential(*features)

    classifier = nn.Sequential(*features)
    return features, classifier


    
if __name__ == '__main__':
    ft, clss = resnet50()
    print(ft)