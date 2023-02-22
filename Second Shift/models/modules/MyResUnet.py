import torch
import torch.nn as nn
import torch.nn.functional as F
from models.modules.modules import spectral_norm
import numpy as np

class MyUnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_spectral_norm=False):
        super(MyUnetGenerator, self).__init__()

        # Encoder layers
        self.e1_c = nn.Conv2d(input_nc, ngf, kernel_size=4, stride=2, padding=1)

        self.e2_c = nn.Conv2d(ngf, ngf*2, kernel_size=4, stride=2, padding=1)
        self.e2_norm = nn.BatchNorm2d(ngf*2)

        self.e3_c = nn.Conv2d(ngf*2, ngf*4, kernel_size=4, stride=2, padding=1)
        self.e3_norm = nn.BatchNorm2d(ngf*4)

        self.e4_c = nn.Conv2d(ngf*4, ngf*8, kernel_size=4, stride=2, padding=1)
        self.e4_norm = nn.BatchNorm2d(ngf*8)

        self.e5_c = nn.Conv2d(ngf*8, ngf*8, kernel_size=4, stride=2, padding=1)
        self.e5_norm = nn.BatchNorm2d(ngf*8)

        self.e6_c = nn.Conv2d(ngf*8, ngf*8, kernel_size=4, stride=2, padding=1)
        self.e6_norm = nn.BatchNorm2d(ngf*8)

        self.e7_c = nn.Conv2d(ngf*8, ngf*8, kernel_size=4, stride=2, padding=1)
        self.e7_norm = nn.BatchNorm2d(ngf*8)

        self.e8_c = nn.Conv2d(ngf*8, ngf*8, kernel_size=4, stride=2, padding=1)

        # Deocder layers
        self.d1_c = nn.ConvTranspose2d(ngf*8, ngf*8, kernel_size=4, stride=2, padding=1)
        self.d1_norm = nn.BatchNorm2d(ngf*8)

        self.d2_c = nn.ConvTranspose2d(ngf*8*2, ngf*8, kernel_size=4, stride=2, padding=1)
        self.d2_norm = nn.BatchNorm2d(ngf*8)

        self.d3_c = nn.ConvTranspose2d(ngf*8*2, ngf*8, kernel_size=4, stride=2, padding=1)
        self.d3_norm = nn.BatchNorm2d(ngf*8)

        self.d4_c = nn.ConvTranspose2d(ngf*8*2, ngf*8, kernel_size=4, stride=2, padding=1)
        self.d4_norm = nn.BatchNorm2d(ngf*8)

        # shift
        self.d5_c = nn.ConvTranspose2d(ngf*8*2, ngf*4, kernel_size=4, stride=2, padding=1)
        self.d5_norm = norm_layer(ngf*4)
        # self.d5_c = nn.ConvTranspose2d(ngf*8*3, ngf*4, kernel_size=4, stride=2, padding=1)
        # self.d5_norm = nn.BatchNorm2d(ngf*4)

        self.d6_c = nn.ConvTranspose2d(ngf*4*2, ngf*2, kernel_size=4, stride=2, padding=1)
        self.d6_norm = nn.BatchNorm2d(ngf*2)

        self.d7_c = nn.ConvTranspose2d(ngf*2*2, ngf, kernel_size=4, stride=2, padding=1)
        self.d7_norm = nn.BatchNorm2d(ngf)

        self.d8_c = nn.ConvTranspose2d(ngf*2, output_nc, kernel_size=4, stride=2, padding=1)

    # In this case, we have very flexible unet construction mode.
    def forward(self, input):
        # Encoder
        # No norm on the first layer
        e1 = self.e1_c(input)
        e2 = self.e2_norm(self.e2_c(F.leaky_relu_(e1, negative_slope=0.2)))
        e3 = self.e3_norm(self.e3_c(F.leaky_relu_(e2, negative_slope=0.2)))
        e4 = self.e4_norm(self.e4_c(F.leaky_relu_(e3, negative_slope=0.2)))
        e5 = self.e5_norm(self.e5_c(F.leaky_relu_(e4, negative_slope=0.2)))
        e6 = self.e6_norm(self.e6_c(F.leaky_relu_(e5, negative_slope=0.2)))
        e7 = self.e7_norm(self.e7_c(F.leaky_relu_(e6, negative_slope=0.2)))
        # No norm on the inner_most layer
        e8 = self.e8_c(F.leaky_relu_(e7, negative_slope=0.2))
        # Decoder
        d1 = self.d1_norm(self.d1_c(F.relu_(e8)))
        d2 = self.d2_norm(self.d2_c(F.relu_(torch.cat([d1, e7], dim=1))))
        d3 = self.d3_norm(self.d3_c(F.relu_(torch.cat([d2, e6], dim=1))))
        d4 = self.d4_norm(self.d4_c(F.relu_(torch.cat([d3, e5], dim=1))))
        # shift
        # d5 = self.d5_norm(self.d5_c(F.relu_(torch.cat([d4, e4, e4], dim=1))))
        d5 = self.d5_norm(self.d5_c(F.relu_(torch.cat([d4, e4], dim=1))))
        d6 = self.d6_norm(self.d6_c(F.relu_(torch.cat([d5, e3], dim=1))))
        d7 = self.d7_norm(self.d7_c(F.relu_(torch.cat([d6, e2], dim=1))))
        # No norm on the last layer
        d8 = self.d8_c(F.relu_(torch.cat([d7, e1], 1)))
        d8 = torch.tanh(d8)

        # print('encode')
        # print(e1.shape)
        # print(e2.shape)
        # print(e3.shape)
        # print(e4.shape)
        # print(e5.shape)
        # print(e6.shape)
        # print(e7.shape)
        # print(e8.shape)
        # print('decode')
        # print(d1.shape)
        # print(d2.shape)
        # print(d3.shape)
        # print(d4.shape)
        # print(d5.shape)
        # print(d6.shape)
        # print(d7.shape)
        # print(d8.shape)
        return d8


class MyGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64):
        super(MyGenerator, self).__init__()

        # pre
        # self.multiconv = MultiScaleConv(input_nc, 16)
        # Encoder layers
        self.e1_c = nn.Conv2d(input_nc, ngf, kernel_size=3, stride=1, padding=1)

        self.e2_c = nn.Conv2d(ngf, ngf*2, kernel_size=3, stride=1, padding=1)
        self.e2_norm = nn.BatchNorm2d(ngf*2)

        self.e3_c = nn.Conv2d(ngf*2, ngf*4, kernel_size=3, stride=1, padding=1)
        self.e3_norm = nn.BatchNorm2d(ngf*4)

        self.e4_c = nn.Conv2d(ngf*4, ngf*8, kernel_size=3, stride=1, padding=1)
        self.e4_norm = nn.BatchNorm2d(ngf*8)

        self.e5_c = nn.Conv2d(ngf*8, ngf*8, kernel_size=3, stride=1, padding=1)
        self.e5_norm = nn.BatchNorm2d(ngf*8)

        self.e6_c = nn.Conv2d(ngf*8, ngf*8, kernel_size=3, stride=1, padding=1)
        self.e6_norm = nn.BatchNorm2d(ngf*8)

        self.e7_c = nn.Conv2d(ngf*8, ngf*8, kernel_size=3, stride=1, padding=1)
        self.e7_norm = nn.BatchNorm2d(ngf*8)

        self.e8_c = nn.Conv2d(ngf*8, ngf*8, kernel_size=3, stride=1, padding=1)

        # Deocder layers
        self.d1_c = nn.ConvTranspose2d(ngf*8, ngf*8, kernel_size=3, stride=1, padding=1)
        self.d1_norm = nn.BatchNorm2d(ngf*8)

        self.d2_c = nn.ConvTranspose2d(ngf*8*2, ngf*8, kernel_size=3, stride=1, padding=1)
        self.d2_norm = nn.BatchNorm2d(ngf*8)

        self.d3_c = nn.ConvTranspose2d(ngf*8*2, ngf*8, kernel_size=3, stride=1, padding=1)
        self.d3_norm = nn.BatchNorm2d(ngf*8)

        self.d4_c = nn.ConvTranspose2d(ngf*8*2, ngf*8, kernel_size=3, stride=1, padding=1)
        self.d4_norm = nn.BatchNorm2d(ngf*8)

        # shift
        self.d5_c = nn.ConvTranspose2d(ngf*8*3, ngf*4, kernel_size=3, stride=1, padding=1)
        self.d5_norm = nn.BatchNorm2d(ngf*4)

        self.d6_c = nn.ConvTranspose2d(ngf*4*2, ngf*2, kernel_size=3, stride=1, padding=1)
        self.d6_norm = nn.BatchNorm2d(ngf*2)

        self.d7_c = nn.ConvTranspose2d(ngf*2*2, ngf, kernel_size=3, stride=1, padding=1)
        self.d7_norm = nn.BatchNorm2d(ngf)

        self.d8_c = nn.ConvTranspose2d(ngf*2, output_nc, kernel_size=3, stride=1, padding=1)

    # In this case, we have very flexible unet construction mode.
    def forward(self, input):
        # Encoder
        # No norm on the first layer

        e1 = self.e1_c(input)
        e2 = self.e2_norm(self.e2_c(F.leaky_relu_(e1, negative_slope=0.2)))
        e3 = self.e3_norm(self.e3_c(F.leaky_relu_(e2, negative_slope=0.2)))
        e4 = self.e4_norm(self.e4_c(F.leaky_relu_(e3, negative_slope=0.2)))
        e5 = self.e5_norm(self.e5_c(F.leaky_relu_(e4, negative_slope=0.2)))
        e6 = self.e6_norm(self.e6_c(F.leaky_relu_(e5, negative_slope=0.2)))
        e7 = self.e7_norm(self.e7_c(F.leaky_relu_(e6, negative_slope=0.2)))
        # No norm on the inner_most layer
        e8 = self.e8_c(F.leaky_relu_(e7, negative_slope=0.2))
        # print('encode')
        # print(e1.shape)
        # print(e2.shape)
        # print(e3.shape)
        # print(e4.shape)
        # print(e5.shape)
        # print(e6.shape)
        # print(e7.shape)
        # print(e8.shape)

        # Decoder
        d1 = self.d1_norm(self.d1_c(F.relu_(e8)))
        d2 = self.d2_norm(self.d2_c(F.relu_(torch.cat([d1, e7], dim=1))))
        d3 = self.d3_norm(self.d3_c(F.relu_(torch.cat([d2, e6], dim=1))))
        d4 = self.d4_norm(self.d4_c(F.relu_(torch.cat([d3, e5], dim=1))))

        # shift
        d5 = self.d5_norm(self.d5_c(F.relu_(torch.cat([d4, e4, e4], dim=1))))

        d6 = self.d6_norm(self.d6_c(F.relu_(torch.cat([d5, e3], dim=1))))
        d7 = self.d7_norm(self.d7_c(F.relu_(torch.cat([d6, e2], dim=1))))
        # No norm on the last layer
        d8 = self.d8_c(F.relu_(torch.cat([d7, e1], 1)))

        d8 = torch.tanh(d8)


        # print('decode')
        # print(d1.shape)
        # print(d2.shape)
        # print(d3.shape)
        # print(d4.shape)
        # print(d5.shape)
        # print(d6.shape)
        # print(d7.shape)
        # print(d8.shape)
        return d8


class MyShiftUnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64):
        super(MyShiftUnetGenerator, self).__init__()

        # Encoder layers
        self.e1_c = nn.Conv2d(input_nc, ngf, kernel_size=4, stride=2, padding=1)

        self.e2_c = nn.Conv2d(ngf, ngf * 2, kernel_size=4, stride=2, padding=1)
        self.e2_norm = nn.BatchNorm2d(ngf * 2)

        self.e3_c = nn.Conv2d(ngf * 2, ngf * 4, kernel_size=4, stride=2, padding=1)
        self.e3_norm = nn.BatchNorm2d(ngf * 4)

        # self.e4_c = nn.Conv2d(ngf * 4, ngf * 8, kernel_size=4, stride=2, padding=1)
        # self.e4_norm = nn.BatchNorm2d(ngf * 8)
        self.res_block = BasicBlock(ngf * 4, ngf * 8)
        self.e4_c = nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1)
        self.e4_norm = nn.BatchNorm2d(ngf * 8)

        self.e5_c = nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1)
        self.e5_norm = nn.BatchNorm2d(ngf * 8)

        self.e6_c = nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1)
        self.e6_norm = nn.BatchNorm2d(ngf * 8)

        self.e7_c = nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1)
        self.e7_norm = nn.BatchNorm2d(ngf * 8)

        self.e8_c = nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1)

        # Deocder layers
        self.d1_c = nn.ConvTranspose2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1)
        self.d1_norm = nn.BatchNorm2d(ngf * 8)

        self.d2_c = nn.ConvTranspose2d(ngf * 8 * 2, ngf * 8, kernel_size=4, stride=2, padding=1)
        self.d2_norm = nn.BatchNorm2d(ngf * 8)

        self.d3_c = nn.ConvTranspose2d(ngf * 8 * 2, ngf * 8, kernel_size=4, stride=2, padding=1)
        self.d3_norm = nn.BatchNorm2d(ngf * 8)

        self.d4_c = nn.ConvTranspose2d(ngf * 8 * 2, ngf * 8, kernel_size=4, stride=2, padding=1)
        self.d4_norm = nn.BatchNorm2d(ngf * 8)

        # shift
        self.d5_c = nn.ConvTranspose2d(ngf*8*2, ngf*4, kernel_size=4, stride=2, padding=1)
        self.d5_norm = nn.BatchNorm2d(ngf*4)
        # self.d5_c = nn.ConvTranspose2d(ngf * 8 * 3, ngf * 4, kernel_size=4, stride=2, padding=1)
        # self.d5_norm = nn.BatchNorm2d(ngf * 4)

        self.d6_c = nn.ConvTranspose2d(ngf * 4 * 2, ngf * 2, kernel_size=4, stride=2, padding=1)
        self.d6_norm = nn.BatchNorm2d(ngf * 2)

        self.d7_c = nn.ConvTranspose2d(ngf * 2 * 2, ngf, kernel_size=4, stride=2, padding=1)
        self.d7_norm = nn.BatchNorm2d(ngf)

        self.d8_c = nn.ConvTranspose2d(ngf * 2, output_nc, kernel_size=4, stride=2, padding=1)

    # In this case, we have very flexible unet construction mode.
    def forward(self, input):
        # Encoder
        # No norm on the first layer
        e1 = self.e1_c(input)
        e2 = self.e2_norm(self.e2_c(F.leaky_relu_(e1, negative_slope=0.2)))
        e3 = self.e3_norm(self.e3_c(F.leaky_relu_(e2, negative_slope=0.2)))
        # e4 = self.e4_norm(self.e4_c(F.leaky_relu_(e3, negative_slope=0.2)))
        e4 = self.e4_norm(self.e4_c(self.res_block(F.leaky_relu_(e3, negative_slope=0.2))))
        e5 = self.e5_norm(self.e5_c(F.leaky_relu_(e4, negative_slope=0.2)))
        e6 = self.e6_norm(self.e6_c(F.leaky_relu_(e5, negative_slope=0.2)))
        e7 = self.e7_norm(self.e7_c(F.leaky_relu_(e6, negative_slope=0.2)))
        # No norm on the inner_most layer
        e8 = self.e8_c(F.leaky_relu_(e7, negative_slope=0.2))

        # Decoder
        d1 = self.d1_norm(self.d1_c(F.relu_(e8)))
        d2 = self.d2_norm(self.d2_c(F.relu_(torch.cat([d1, e7], dim=1))))
        d3 = self.d3_norm(self.d3_c(F.relu_(torch.cat([d2, e6], dim=1))))
        d4 = self.d4_norm(self.d4_c(F.relu_(torch.cat([d3, e5], dim=1))))

        # shift
        d5 = self.d5_norm(self.d5_c(F.relu_(torch.cat([d4, e4], dim=1))))
        # d5 = self.d5_norm(self.d5_c(F.relu_(torch.cat([d4, e4, e4], dim=1))))

        d6 = self.d6_norm(self.d6_c(F.relu_(torch.cat([d5, e3], dim=1))))
        d7 = self.d7_norm(self.d7_c(F.relu_(torch.cat([d6, e2], dim=1))))
        # No norm on the last layer
        d8 = self.d8_c(F.relu_(torch.cat([d7, e1], 1)))

        d8 = torch.tanh(d8)

        # print('encode')
        # print(e1.shape)
        # print(e2.shape)
        # print(e3.shape)
        # print(e4.shape)
        # print(e5.shape)
        # print(e6.shape)
        # print(e7.shape)
        # print(e8.shape)
        # print('decode')
        # print(d1.shape)
        # print(d2.shape)
        # print(d3.shape)
        # print(d4.shape)
        # print(d5.shape)
        # print(d6.shape)
        # print(d7.shape)
        # print(d8.shape)
        return d8


class _Transition(nn.Sequential):
    """Transition layer between two adjacent DenseBlock"""
    def __init__(self, num_input_feature, num_output_features):
        super(_Transition, self).__init__()
        self.add_module("norm", nn.BatchNorm2d(num_input_feature))
        self.add_module("relu", nn.ReLU(inplace=True))
        self.add_module("conv", nn.Conv2d(num_input_feature, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module("pool", nn.AvgPool2d(2, stride=2))


class _DenseBlock(nn.Sequential):
    """DenseBlock"""
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features+i*growth_rate, growth_rate, bn_size,
                                drop_rate)
            self.add_module("denselayer%d" % (i+1,), layer)


class _DenseLayer(nn.Sequential):
    """Basic unit of DenseBlock (using bottleneck layer) """
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module("norm1", nn.BatchNorm2d(num_input_features))
        self.add_module("relu1", nn.ReLU(inplace=True))
        self.add_module("conv1", nn.Conv2d(num_input_features, bn_size*growth_rate,
                                           kernel_size=1, stride=1, bias=False))
        self.add_module("norm2", nn.BatchNorm2d(bn_size*growth_rate))
        self.add_module("relu2", nn.ReLU(inplace=True))
        self.add_module("conv2", nn.Conv2d(bn_size*growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1, bias=False))
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34
       """
    # BasicBlock and BottleNeck block
    # have different output size
    # we use class attribute expansion
    # to distinct
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()

       # residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )

        # shortcut
        self.shortcut = nn.Sequential()

        # the shortcut output dimension is not the same with residual function
        # use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class BottleNeck(nn.Module):
    """Residual block for resnet over 50 layers
    """
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class DP_Conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DP_Conv, self).__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=in_ch
        )
        self.point_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1
        )

    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out


class MultiScaleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MultiScaleConv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, stride=1, kernel_size=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(in_channels, out_channels, stride=1, kernel_size=3, padding=1, bias=False)
        self.conv3 = nn.Conv2d(in_channels, out_channels, stride=1, kernel_size=5, padding=2, bias=False)
        self.conv4 = nn.Conv2d(in_channels, out_channels, stride=1, kernel_size=7, padding=3, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x1 = self.bn(self.conv1(x))
        x2 = self.bn(self.conv2(x))
        x3 = self.bn(self.conv3(x))
        x4 = self.bn(self.conv4(x))
        out = torch.cat([x1, x2, x3, x4], dim=1)
        return out


class MyTest(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64):
        super(MyTest, self).__init__()

        # pre
        # self.multiconv = MultiScaleConv(input_nc, 16)
        # Encoder layers
        # self.e1_c = nn.Conv2d(input_nc, ngf, kernel_size=4, stride=2, padding=1)
        #
        # self.e2_c = nn.Conv2d(ngf, ngf * 2, kernel_size=4, stride=2, padding=1)
        self.e1_c = DP_Conv(input_nc, ngf)
        self.e2_c = DP_Conv(ngf, ngf * 2)
        # self.e2_c = nn.Conv2d(ngf, ngf * 2, kernel_size=4, stride=2, padding=1)
        self.e2_norm = nn.BatchNorm2d(ngf * 2)

        #  self.e3_c = DP_Conv(ngf * 2, ngf * 4)
        self.e3_c = nn.Conv2d(ngf * 2, ngf * 4, kernel_size=4, stride=2, padding=1)
        self.e3_norm = nn.BatchNorm2d(ngf * 4)

        # self.e4_c = DP_Conv(ngf * 4, ngf * 8)
        self.e4_c = nn.Conv2d(ngf * 4, ngf * 8, kernel_size=4, stride=2, padding=1)
        self.e4_norm = nn.BatchNorm2d(ngf * 8)

        # self.e5_c = DP_Conv(ngf * 8, ngf * 8)
        self.e5_c = nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1)
        self.e5_norm = nn.BatchNorm2d(ngf * 8)

        self.e6_c = nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1)
        self.e6_norm = nn.BatchNorm2d(ngf * 8)

        self.e7_c = nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1)
        self.e7_norm = nn.BatchNorm2d(ngf * 8)

        self.e8_c = nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1)

        # Deocder layers
        self.d1_c = nn.ConvTranspose2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1)
        self.d1_norm = nn.BatchNorm2d(ngf * 8)

        self.d2_c = nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1)
        self.d2_norm = nn.BatchNorm2d(ngf * 8)

        self.d3_c = nn.ConvTranspose2d(1024, ngf * 8, kernel_size=4, stride=2, padding=1)
        self.d3_norm = nn.BatchNorm2d(ngf * 8)

        self.d4_c = nn.ConvTranspose2d(1024, ngf * 8, kernel_size=4, stride=2, padding=1)
        self.d4_norm = nn.BatchNorm2d(ngf * 8)

        # shift
        # self.d5_c = spectral_norm(nn.ConvTranspose2d(ngf*8*2, ngf*4, kernel_size=4, stride=2, padding=1), use_spectral_norm)
        # self.d5_norm = norm_layer(ngf*4)
        self.d5_c = nn.ConvTranspose2d(ngf * 8 * 3, ngf * 4, kernel_size=4, stride=2, padding=1)
        self.d5_norm = nn.BatchNorm2d(ngf * 4)

        self.d6_c = nn.ConvTranspose2d(ngf * 4 * 2, ngf * 2, kernel_size=4, stride=2, padding=1)
        self.d6_norm = nn.BatchNorm2d(ngf * 2)

        self.d7_c = nn.ConvTranspose2d(256, ngf, kernel_size=3, stride=1, padding=1)
        self.d7_norm = nn.BatchNorm2d(ngf)

        self.d8_c = nn.ConvTranspose2d(ngf * 2, output_nc, kernel_size=3, stride=1, padding=1)

    def forward(self, input):
        # Encoder
        # No norm on the first layer
        e1 = self.e1_c(input)
        # e2 = self.e2_norm(self.e2_c(F.leaky_relu_(e1, negative_slope=0.2)))
        # e3 = self.e3_norm(self.e3_c(F.leaky_relu_(e2, negative_slope=0.2)))
        # e4 = self.e4_norm(self.e4_c(F.leaky_relu_(e3, negative_slope=0.2)))
        e2 = self.e2_c(F.leaky_relu_(e1, negative_slope=0.2))
        e3 = self.e3_c(F.leaky_relu_(e2, negative_slope=0.2))
        e4 = self.e4_c(F.leaky_relu_(e3, negative_slope=0.2))

        e5 = self.e5_c(F.leaky_relu_(e4, negative_slope=0.2))
        e6 = self.e6_c(F.leaky_relu_(e5, negative_slope=0.2))
        e7 = self.e7_c(F.leaky_relu_(e6, negative_slope=0.2))
        # No norm on the inner_most layer
        e8 = self.e8_c(F.leaky_relu_(e7, negative_slope=0.2))
        # print('encode')
        # print(e1.shape)
        # print(e2.shape)
        # print(e3.shape)
        # print(e4.shape)
        # print(e5.shape)
        # print(e6.shape)
        # print(e7.shape)
        # print(e8.shape)

        # Decoder
        d1 = self.d1_norm(self.d1_c(F.relu_(e8)))
        # print('decode')
        # print(d1.shape)
        d2 = self.d2_norm(self.d2_c(F.relu_(torch.cat([d1, e7], dim=1))))
        d3 = self.d3_norm(self.d3_c(F.relu_(torch.cat([d2, e6], dim=1))))
        d4 = self.d4_norm(self.d4_c(F.relu_(torch.cat([d3, e5], dim=1))))

        # shift
        d5 = self.d5_norm(self.d5_c(F.relu_(torch.cat([d4, e4, e4], dim=1))))
        d6 = self.d6_norm(self.d6_c(F.relu_(torch.cat([d5, e3], dim=1))))
        d7 = self.d7_norm(self.d7_c(F.relu_(torch.cat([d6, e2], dim=1))))

        # print(d2.shape)
        # print(d3.shape)
        # print(d4.shape)
        # print(d5.shape)
        # print(d6.shape)
        # print(d7.shape)
        # No norm on the last layer
        d8 = self.d8_c(F.relu_(torch.cat([d7, e1], 1)))

        d8 = torch.tanh(d8)


        # print(d2.shape)
        # print(d3.shape)
        # print(d4.shape)
        # print(d5.shape)
        # print(d6.shape)
        # print(d7.shape)
        # print(d8.shape)
        return d8


if __name__ == '__main__':
    x = torch.randn(1, 3, 256, 256)
    net = MyShiftUnetGenerator(3, 3)
    out = net(x)
    print('out.shape: ', out.shape)

    # 定义总参数量、可训练参数量及非可训练参数量变量
    Total_params = 0
    Trainable_params = 0
    NonTrainable_params = 0
    model = net
    # 遍历model.parameters()返回的全局参数列表
    for param in model.parameters():
        mulValue = np.prod(param.size())  # 使用numpy prod接口计算参数数组所有元素之积
        Total_params += mulValue  # 总参数量
        if param.requires_grad:
            Trainable_params += mulValue  # 可训练参数量
        else:
            NonTrainable_params += mulValue  # 非可训练参数量

    print(f'Total params: {Total_params}')
    print(f'Trainable params: {Trainable_params}')
    print(f'Non-trainable params: {NonTrainable_params}')
