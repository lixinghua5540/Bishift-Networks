import cv2
import os
import skimage
import numpy as np
import util
import html
import time
from subprocess import Popen, PIPE
import sys
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models import create_model
import visdom
import torch
from torch import nn
from models.patch_soft_shift.innerPatchSoftShiftTripleModule import InnerPatchSoftShiftTripleModule
from torchvision import transforms
from torchvision import models
from natsort import natsorted


def toTensor(img):
    assert type(img) == np.ndarray, 'the img type is {}, but ndarry expected'.format(type(img))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img.transpose((2, 0, 1)))
    return img.float().div(255).unsqueeze(0)  # 255也可以改为256


def double_conv(in_channels, out_channels):  # 双层卷积模型，神经网络最基本的框架
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),  # 加入Bn层提高网络泛化能力（防止过拟合），加收敛速度
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),  # 3指kernel_size，即卷积核3*3
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


class UNet(nn.Module):
    def __init__(self, n_class):
        super().__init__()

        self.dconv_down1 = double_conv(3, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dconv_up3 = double_conv(256 + 512, 256)  # torch.cat后输入深度变深
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(128 + 64, 64)

        self.conv_last = nn.Conv2d(64, n_class, 1)

    def forward(self, x):
        # encode
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)

        x = self.dconv_down4(x)

        # decode
        x = self.upsample(x)
        # 因为使用了3*3卷积核和 padding=1 的组合，所以卷积过程图像尺寸不发生改变，所以省去了crop操作！
        x = torch.cat([x, conv3], dim=1)

        x = self.dconv_up3(x)
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up2(x)
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)

        x = self.dconv_up1(x)

        out = self.conv_last(x)

        return out


if __name__ == "__main__":
    # path1 = os.path.abspath('..')  # 获取上一级目录
    # print()
    # images = []
    # dir = 'results/exp/test_50/images'
    # assert os.path.isdir(dir), '%s is not a valid directory' % dir
    # for root, _, fnames in sorted(os.walk(dir)):
    #     for fname in fnames:
    #             path = fname
    #             images.append(path)
    # print(images)
    # print(natsorted(images))
    # nn.Conv2d(in_channels=4, out_channels=64, stride=1, kernel_size=3, padding=0, dilation=2)
    import torch

    print(torch.cuda.is_available())

