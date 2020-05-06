import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
import sys
import os
import shutil

import deform_conv

f1 = open("output-resnet-size.txt", "a")

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)

#def conv_init(m):
#    classname = m.__class__.__name__
#    if classname.find('Conv') != -1:
#        init.xavier_uniform(m.weight, gain=np.sqrt(2))
#        init.constant(m.bias, 0)

def cfg(depth):
    depth_lst = [18, 34, 50, 101, 152]
    assert (depth in depth_lst), "Error : Resnet depth should be either 18, 34, 50, 101, 152"
    cf_dict = {
        '18': (BasicBlock, [2,2,2,2]),
        '34': (BasicBlock, [3,4,6,3]),
        '50': (Bottleneck, [3,4,6,3]),
        '101':(Bottleneck, [3,4,23,3]),
        '152':(Bottleneck, [3,8,36,3]),
    }

    return cf_dict[str(depth)]

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=True),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=True)
        self.bn1 = nn.BatchNorm2d(planes)
        #apply deformable layer here as specified in the paper
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)
        #self.conv2 = deform_conv.DeformConv2D(planes, planes, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=True)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        #if stride used is not 1 (but 2 like in layer 2 and 3), 
        #then need to perform convolution of 1 x 1 first before shortcut connection
        #OR if input filter (in_planes) not equal to output filter (self.expansion*planes), 
        #then also need to perform convolution of 1 x 1 first before shortcut connection to make the
        #dimension match in summation
        if stride != 1 or in_planes != self.expansion*planes:
            print('stride: ', stride)
            print('in_planes: ', in_planes, '   self.expansion*planes: ', self.expansion*planes)
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=True),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        print('*************************START******************************', file = f1)
        print('x: ', x.size(), file = f1)
        out = F.relu(self.bn1(self.conv1(x)))
        print('out1: ', out.size(), file = f1)
        out = F.relu(self.bn2(self.conv2(out)))
        print('out2: ', out.size(), file = f1)
        out = self.bn3(self.conv3(out))
        print('out3: ', out.size(), file = f1)
        print('x2: ', x.size(), file = f1)
        out += self.shortcut(x)
        print('out4: ', out.size(), file = f1)
        print('x3: ', x.size(), file = f1)
        out = F.relu(out)
        print('out5: ', out.size(), file = f1)
        print('***************************END******************************', file = f1)

        return out

class ResNet(nn.Module):
    def __init__(self, depth, num_classes):
        super(ResNet, self).__init__()
        self.in_planes = 16

        block, num_blocks = cfg(depth)
        
        print('depth: ', depth)
        print('block: ', block)
        print('num_blocks: ', num_blocks)

        
        self.conv1 = conv3x3(3,16)
       
        self.bn1 = nn.BatchNorm2d(16)
        
        
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
         
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
         
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        
        self.linear = nn.Linear(64*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        print('START RESNET 101', file = f1)
        print('input x to network: ', x.size(), file = f1)
        out = F.relu(self.bn1(self.conv1(x)))
        print('out1 resnet: ', out.size(), file = f1)
        print('START LAYER 1 WITH 16 FILTER', file = f1)
        out = self.layer1(out)
        print('out2 from layer 1: ', out.size(), file = f1)
        print('START LAYER 2 WITH 32 FILTER', file = f1)
        out = self.layer2(out)
        print('out3 from layer 2: ', out.size(), file = f1)
        print('START LAYER 3 WITH 64 FILTER', file = f1)
        out = self.layer3(out)
        print('out4 from layer 3: ', out.size(), file = f1)
        out = F.avg_pool2d(out, 8) 
        print('out5 from avg pool: ', out.size(), file = f1)
        out = out.view(out.size(0), -1)
        print('out6 from view: ', out.size(), file = f1)
        out = self.linear(out)
        print('out7 from linear: ', out.size(), file = f1)

        return out
    
def ResNet101():
    return ResNet(101, num_classes = 10)

if __name__ == '__main__':
    net=ResNet101()
    print(net)
    #y = net(Variable(torch.randn(1,3,32,32)))
    #print(y.size())
