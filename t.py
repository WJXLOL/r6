import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import os
import torch.nn.init as init
from PIL import Image
import numpy as np
from torchvision.datasets import ImageFolder
import torchvision


def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class ResNet_modify(nn.Module):

    def __init__(self, block, num_blocks, num_classes=100, nf=64):
        super(ResNet_modify, self).__init__()
        self.in_planes = nf

        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.layer1 = self._make_layer(block, 1 * nf, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 2 * nf, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 4 * nf, num_blocks[2], stride=2)
        self.out_dim = 4 * nf * block.expansion

        self.fc = nn.Linear(self.out_dim, num_classes)
        # self.fc_cb = torch.nn.utils.weight_norm(nn.Linear(512 * block.expansion, num_class), dim=0)
        hidden_dim = 128
        self.fc_cb = nn.Linear(self.out_dim, num_classes)
        self.contrast_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.projection_head = nn.Sequential(
            nn.Linear(self.out_dim, hidden_dim),

        )
        self.apply(_weights_init)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 5, stride=2, padding=1),  # [batch, 24, 8, 8]
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1),  # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.ConvTranspose2d(64, 3, 2, stride=1, padding=1),  # [batch, 3, 32, 32]
            nn.Sigmoid(),
        )

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x, train=False):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        z = out

        out = F.avg_pool2d(out, out.size()[3])
        feature = out.view(out.size(0), -1)

        if train is True:
            out = self.fc(feature)
            out_cb = self.fc_cb(feature)
            z = self.projection_head(feature)
            p = self.contrast_head(z)
            return out, out_cb, z, p
        else:
            out = self.fc_cb(feature)

            return out, z


class LambdaLayer(nn.Module):

    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock_s(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock_s, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes // 4, planes // 4), "constant",
                                                  0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


def ResNet32(num_class=100):
    return ResNet_modify(BasicBlock_s, [5, 5, 5], num_classes=num_class)


# 用上采样加卷积代替了反卷积
class ResizeConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, mode='nearest'):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=1, bias=False)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        x = self.conv(x)
        return x


class BasicBlockDec(nn.Module):
    expansion = 1

    def __init__(self, in_planes, stride=1):
        super().__init__()

        planes = int(in_planes / stride)

        self.conv2 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_planes)

        if stride == 1:
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential()
        else:
            self.conv1 = ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride)
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential(
                ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn2(self.conv2(x)))
        out = self.bn1(self.conv1(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out


class Dec(nn.Module):

    def __init__(self, num_Blocks=[3, 3, 3], z_dim=32, nc=3, nf=64):
        super().__init__()
        self.in_planes = nf
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.layer4 = self._make_layer2(BasicBlockDec, 4 * nf, num_Blocks[2], stride=2)
        self.layer5 = self._make_layer2(BasicBlockDec, 2 * nf, num_Blocks[1], stride=2)
        self.layer6 = self._make_layer2(BasicBlockDec, 1 * nf, num_Blocks[0], stride=1)
        self.conv2 = ResizeConv2d(nf, nc, kernel_size=3, scale_factor=1)
        #self.convt = nn.ConvTranspose2d(64, 3, 2, stride=1, padding=1)
    def _make_layer2(self, BasicBlockDec, planes, num_Blocks, stride):
        strides = [stride] + [1] * (num_Blocks - 1)
        layers = []
        for stride in reversed(strides):
            self.in_planes = planes * BasicBlockDec.expansion
            layers += [BasicBlockDec(self.in_planes, stride)]

        return nn.Sequential(*layers)

    def forward(self, z):
        # x = self.linear(z)
        # x = x.view(z.size(0), 512, 1, 1)
        # x = F.interpolate(x, scale_factor=7)
        x = self.layer3(z)
        x = self.layer2(x)
        x = self.layer1(x)
        # x = F.interpolate(x, size=(112, 112), mode='bilinear')
        x = torch.sigmoid(self.bn1(x))
        x = self.conv1(x)
        x = x.view(x.size(0), 3, 32, 32)
        return x



if __name__ == "__main__":
    gpu = 0
    device = 'cuda:' + str(gpu) if torch.cuda.is_available() else 'cpu'
    decoder = Dec().to(device)
    encoder = ResNet32().to(device)
    a = torch.randn(64, 3, 32, 32, device='cuda:0')
    _, b = encoder.forward(a)
    c = decoder.forward(b)
    criterion2 = nn.MSELoss().to(device)
    print(criterion2(a, c))
