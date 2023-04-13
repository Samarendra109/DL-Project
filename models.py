import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x, train=True):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        if not train:
            return None, out
        else:
            out = F.relu(out)
            return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, temp=1.0):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.fc = nn.Linear(512*block.expansion, num_classes)
        self.temp = temp

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, k=0, train=True):
        '''
        :param x:
        :param k:
        :param train: switch model to test and extract the FMs of the kth layer
        :return:
        '''
        i = 0
        out = self.bn1(self.conv1(x))
        if k==i and not(train):
            return None, out.view(out.shape[0], -1)
        out = F.relu(out)
        i +=1
        for module in self.layer1:
            if k ==i and not(train):
                _, out = module(out, train=False)    # take the output of ResBlock before relu
                return None, out.view(out.shape[0], -1)
            else:
                out = module(out)
            out = F.relu(out)
            i+=1

        for module in self.layer2:
            if k ==i and not(train):
                _, out = module(out, train=False)    # take the output of ResBlock before relu
                return None, out.view(out.shape[0], -1)
            else:
                out = module(out)
            out = F.relu(out)
            i+=1
        for module in self.layer3:
            if k ==i and not(train):
                _, out = module(out, train=False)    # take the output of ResBlock before relu
                return None, out.view(out.shape[0], -1)
            else:
                out = module(out)
            out = F.relu(out)
            i+=1
        for module in self.layer4:
            if k ==i and not(train):
                _, out = module(out, train=False)    # take the output of ResBlock before relu
                return None, out.view(out.shape[0], -1)
            else:
                out = module(out)
            out = F.relu(out)
            i+=1
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out) / self.temp
        if k == i and not (train):
            _f = F.softmax(out, 1)  # take the output of softmax
            return None, _f
        else:
            return out