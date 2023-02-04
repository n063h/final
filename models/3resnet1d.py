from .context_block import ContextBlock
from torch.nn.utils.weight_norm import WeightNorm
import torch
import os
import sys
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math

from utils.base import Config
# from dcn.deform_conv import ModulatedDeformConvPack
sys.path.append('/home/lvhang/Tricks-of-Semi-supervisedDeepLeanring-Pytorch/architectures')


class distLinear(nn.Module):
    def __init__(self, indim, outdim):
        super(distLinear, self).__init__()
        self.L = nn.Linear(indim, outdim, bias=False)
        self.class_wise_learnable_norm = True  # See the issue#4&8 in the github
        if self.class_wise_learnable_norm:
            # split the weight update component to direction and norm
            WeightNorm.apply(self.L, 'weight', dim=0)

        if outdim <= 200:
            self.scale_factor = 2  # a fixed scale factor to scale the output of cos value into a reasonably large input for softmax, for to reproduce the result of CUB with ResNet10, use 4. see the issue#31 in the github
        else:
            # in omniglot, a larger scale factor is required to handle >1000 output classes.
            self.scale_factor = 10

    def forward(self, x):

        # x = x.view()
        # print(x.size())
        x_norm = torch.norm(x, p=2, dim=1).unsqueeze(1).expand_as(x)
        x_normalized = x.div(x_norm + 0.00001)
        if not self.class_wise_learnable_norm:
            L_norm = torch.norm(self.L.weight.data, p=2, dim=1).unsqueeze(
                1).expand_as(self.L.weight.data)
            self.L.weight.data = self.L.weight.data.div(L_norm + 0.00001)
        # matrix product by forward function, but when using WeightNorm, this also multiply the cosine distance by a class-wise learnable norm, see the issue#4&8 in the github
        cos_dist = self.L(x_normalized)
        scores = self.scale_factor * (cos_dist)
        # print(scores)

        return scores


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    # return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
    #                  padding=dilation, groups=groups, bias=False, dilation=dilation)

    # TODO change to 1d
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


# def conv1x1(in_planes, out_planes, stride=1):
#     """1x1 convolution"""
#     # return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
#     # TODO change to 1d
#     return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    # TODO change to 1d
    return nn.Conv1d(in_planes, out_planes, kernel_size=11, stride=stride, bias=False, padding=5)


def ConvBNReLU(in_channels, out_channels, kernel_size, stride=1, padding=0):
    return nn.Sequential(
        nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                  kernel_size=kernel_size, stride=stride, padding=padding),
        nn.BatchNorm1d(out_channels),
        nn.ReLU6(inplace=True),
    )


class ResNet_d(nn.Module):

    def __init__(self, block, layers, zero_init_residual=False,
                 groups=1, width_per_group=61, in_ch=228, replace_stride_with_dilation=None,
                 norm_layer=None,num_classes=50):
        super(ResNet_d, self).__init__()
        if norm_layer is None:
            # norm_layer = nn.BatchNorm2d
            norm_layer = nn.BatchNorm1d
        self._norm_layer = norm_layer

        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.inplanes = 8*8
        self.groups = groups
        self.base_width = width_per_group
        # TODO
        self.conv1 = nn.Conv1d(in_ch, self.inplanes, kernel_size=11, stride=1, padding=5,
                               bias=False)

        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        layer_channels = [8*8, 16*8, 32*8, 64*8]

        self.refine1 = ContextBlock(inplanes=8*8 * 114, ratio=1. / 4., )
        self.refine2 = ContextBlock(inplanes=16*8 * 57, ratio=1. / 4., )
        self.refine3 = ContextBlock(inplanes=32*8 * 29, ratio=1. / 4., )
        self.refine4 = ContextBlock(inplanes=64*8 * 15, ratio=1. / 4., )
        #
        # self.avgpool1 = nn.AvgPool1d(16)
        # self.avgpool2 = nn.AvgPool1d(8)
        # self.avgpool3 = nn.AvgPool1d(4)
        # self.avgpool4 = nn.AvgPool1d(2)
        #
        # self.s1 = nn.Conv1d(64, 64, 100)
        # self.s2 = nn.Conv1d(128, 128, 43)
        # self.s3 = nn.Conv1d(256, 256, 15)
        # self.s4 = nn.Conv1d(512, 512, 1)

        self.layer1 = self._make_layer(
            block, layer_channels[0], layers[0], stride=1)

        self.layer2 = self._make_layer(block, layer_channels[1], layers[1], stride=1,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, layer_channels[2], layers[2], stride=1,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, layer_channels[3], layers[3], stride=1,
                                       dilate=replace_stride_with_dilation[2])

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.down = nn.Linear(6912, 256)

        # self.fc1 = nn.Linear(2304, 256)
        self.fc1 = nn.Linear(6912, 256)
        self.relu2 = nn.ReLU(inplace=True)
        self.dp=nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(256, num_classes)

        # self.dropout1 = nn.Dropout(p=0.3)

        self.soft1 = nn.Softmax(dim=1)

        self.branch1 = ConvBNReLU(
            in_channels=64*8, out_channels=64*8, kernel_size=1)

        self.branch2 = nn.Sequential(
            ConvBNReLU(in_channels=64*8, out_channels=96*8, kernel_size=1),
            ConvBNReLU(in_channels=96*8, out_channels=128 *
                       8, kernel_size=3, padding=1),
        )

        self.branch3 = nn.Sequential(
            ConvBNReLU(in_channels=64*8, out_channels=16*8, kernel_size=1),
            ConvBNReLU(in_channels=16*8, out_channels=32 *
                       8, kernel_size=3, padding=1),
            ConvBNReLU(in_channels=32*8, out_channels=64 *
                       8, kernel_size=3, padding=1),
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            ConvBNReLU(in_channels=64*8, out_channels=32*8, kernel_size=1),
        )

        self.media = ConvBNReLU(
            in_channels=64*8, out_channels=64*8, kernel_size=7, stride=3)


        for m in self.modules():
            # if isinstance(m, nn.Conv2d):
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            # elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes, stride),
                norm_layer(planes),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, norm_layer))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.view([int(x.size(dim=0)), 1, 228]).float()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)

        b = x
        x = x.view(int(x.size(dim=0)), int(
            x.size(dim=1)) * int(x.size(dim=2)), 1, 1)
        x = self.refine1(x)

        x = x.view(int(b.size(dim=0)), int(b.size(dim=1)), int(b.size(dim=2)))

        x = self.layer2(x)
        a = x
        x = x.view(int(x.size(dim=0)), int(
            x.size(dim=1)) * int(x.size(dim=2)), 1, 1)
        x = self.refine2(x)
        x = x.view(int(a.size(dim=0)), int(a.size(dim=1)), int(a.size(dim=2)))

        # s2 = self.avgpool2(x)
        # s2 = self.s2(x)

        x = self.layer3(x)
        c = x
        x = x.view(int(x.size(dim=0)), int(
            x.size(dim=1)) * int(x.size(dim=2)), 1, 1)
        x = self.refine3(x)
        x = x.view(int(c.size(dim=0)), int(c.size(dim=1)), int(c.size(dim=2)))

        x = self.layer4(x)
        d = x
        x = x.view(int(x.size(dim=0)), int(
            x.size(dim=1)) * int(x.size(dim=2)), 1, 1)
        x = self.refine4(x)
        x = x.view(int(d.size(dim=0)), int(d.size(dim=1)), int(d.size(dim=2)))

        # s4 = self.avgpool4(x)
        # s4 = self.s4(x)

        # x = torch.cat([s1, s2, s3, s4], dim=1)

        x = self.media(x)

        # print(x.shape)

        b_1 = self.branch1(x)
        b_2 = self.branch2(x)
        b_3 = self.branch3(x)
        b_4 = self.branch4(x)
        x = torch.cat([b_1, b_2, b_3, b_4], dim=1)
        # print(x.shape)
        x = x.view(x.size(0), -1)
        embd = x
        # print(embd.shape)

        # print(embd.size())
        # embd = self.down(embd)

        x = self.fc1(embd)
        x = self.relu2(x)
        
        x = self.dp(x)

        out = self.fc3(x)

        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            # norm_layer = nn.BatchNorm2d
            norm_layer = nn.BatchNorm1d
        # TODO
        # if groups != 1 or base_width != 64:
        #     raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)

        # self.conv2 = ModulatedDeformConvPack(planes, planes, kernel_size=1, stride=1)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        # self.channelpool = nn.Conv1d(planes*2, planes,1)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # a = out
        # out = out.view(int(out.size(dim=0)), int(out.size(dim=1)) , (int(out.size(dim=2))), 1)

        out = self.conv2(out)

        # out = out.view(int(a.size(dim=0)),  int(a.size(dim=1)), int(a.size(dim=2)))

        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class BasicBlock_d(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, norm_layer=None):
        super(BasicBlock_d, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, planes, 1)
        self.bn1 = norm_layer(planes)
        self.relu = self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(
            planes, planes, kernel_size=11, stride=2, padding=5)
        self.bn2 = norm_layer(planes)
        self.downsample = nn.Conv1d(inplanes, planes, kernel_size=1, stride=2)
        self.stride = stride

    def forward(self, x):
        identity = x
        #print("basic block",x.shape)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # print("out1:",out.shape)

        out = self.conv2(out)
        out = self.bn2(out)
        # print("out2:",out.shape)

        # if self.downsample is not None:
        identity = self.downsample(identity)

        out += identity
        # print('+:',out.shape)
        out = self.relu(out)

        return out


class BasicBlock_c(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock_c, self).__init__()
        if norm_layer is None:
            # norm_layer = nn.BatchNorm2d
            norm_layer = nn.BatchNorm1d
        # TODO
        # if groups != 1 or base_width != 64:
        #     raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, planes, 1)
        self.bn1 = norm_layer(planes)
        self.relu = self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv1x1(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        # self.channelpool = nn.Conv1d(planes*2, planes,1)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            # norm_layer = nn.BatchNorm2d
            norm_layer = nn.BatchNorm1d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet_c(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=61, in_ch=5, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet_c, self).__init__()
        if norm_layer is None:
            # norm_layer = nn.BatchNorm2d
            norm_layer = nn.BatchNorm1d
        self._norm_layer = norm_layer

        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        # self.inplanes = 64
        self.inplanes = 256
        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group
        # self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
        #                        bias=False)
        # TODO
        self.conv1 = nn.Conv1d(in_ch, self.inplanes, kernel_size=1, stride=1, padding=0,
                               bias=False)

        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        layer_channels = [256, 512, 1024, 2048]
        # layer_channels = [512, 1024, 2048, 4096]

        self.refine_initial = ContextBlock(
            inplanes=228*2+228+150+100+80+60+40+20, ratio=1. / 4., )

        self.refine1 = ContextBlock(inplanes=256, ratio=1. / 4., )
        self.refine2 = ContextBlock(inplanes=512, ratio=1. / 4., )
        self.refine3 = ContextBlock(inplanes=1024, ratio=1. / 4., )
        self.refine4 = ContextBlock(inplanes=2048, ratio=1. / 4., )

        # self.refine1 = ContextBlock(inplanes=512, ratio=1. / 4., )
        # self.refine2 = ContextBlock(inplanes=1024, ratio=1. / 4., )
        # self.refine3 = ContextBlock(inplanes=2048, ratio=1. / 4., )
        # self.refine4 = ContextBlock(inplanes=4096, ratio=1. / 4., )
        # layer_channels = [256, 512, 1024, 2048]
        self.layer1 = self._make_layer(block, layer_channels[0], layers[0])
        self.layer2 = self._make_layer(block, layer_channels[1], layers[1], stride=1,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, layer_channels[2], layers[2], stride=1,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, layer_channels[3], layers[3], stride=1,
                                       dilate=replace_stride_with_dilation[2])

        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        # self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.fc1 = nn.Linear(2048, 256)
        # self.fc1soft1 = nn.Softmax(dim=1)
        self.fc11 = nn.Linear(256, 2)
        #

        self.fc2 = nn.Linear(2048, 256)
        self.fc22 = nn.Linear(256, 2)

        self.fc3 = nn.Linear(2048, 256)
        self.fc33 = nn.Linear(256, 2)

        self.fc4 = nn.Linear(2048, 256)
        self.fc44 = nn.Linear(256, 2)

        self.fc5 = nn.Linear(2048, 256)
        self.fc55 = nn.Linear(256, 2)

        self.fc6 = nn.Linear(2048, 256)
        self.fc66 = nn.Linear(256, 2)

        self.fc7 = nn.Linear(2048, 256)
        self.fc77 = nn.Linear(256, 2)

        self.fc8 = nn.Linear(2048, 256)
        self.fc88 = nn.Linear(256, 2)

        self.fc9 = nn.Linear(2048, 256)
        self.fc99 = nn.Linear(256, 2)

        self.fc9 = nn.Linear(2048, 256)
        self.fc99 = nn.Linear(256, 2)

        self.fc10 = nn.Linear(2048, 256)
        self.fc1010 = nn.Linear(256, 2)

        self.fc111 = nn.Linear(2048, 256)
        self.fc1111 = nn.Linear(256, 2)

        self.fc12 = nn.Linear(2048, 256)
        self.fc1212 = nn.Linear(256, 2)

        self.fc13 = nn.Linear(2048, 256)
        self.fc1313 = nn.Linear(256, 2)

        #

        self.soft1 = nn.Softmax(dim=1)
        self.soft2 = nn.Softmax(dim=1)
        self.soft3 = nn.Softmax(dim=1)
        self.soft4 = nn.Softmax(dim=1)
        self.soft5 = nn.Softmax(dim=1)
        self.soft6 = nn.Softmax(dim=1)
        self.soft7 = nn.Softmax(dim=1)
        self.soft8 = nn.Softmax(dim=1)
        self.soft9 = nn.Softmax(dim=1)
        self.soft10 = nn.Softmax(dim=1)
        self.soft11 = nn.Softmax(dim=1)
        self.soft12 = nn.Softmax(dim=1)
        self.soft13 = nn.Softmax(dim=1)

        # self.upsample = nn.Linear(228, 228*2)
        # self.downs    = nn.Linear(228*2, 57)

        self.initial_fe1 = nn.Linear(228, 228)
        self.initial_bn1 = norm_layer(228)
        self.initial_relu1 = nn.ReLU(inplace=True)
        self.initial_fe2 = nn.Linear(228, 228)
        self.initial_bn2 = norm_layer(228)
        self.initial_relu2 = nn.ReLU(inplace=True)
        self.initial_fe3 = nn.Linear(228, 228)
        self.initial_bn3 = norm_layer(228)
        self.initial_relu3 = nn.ReLU(inplace=True)

        self.scale1 = nn.Linear(228, 228 * 2)
        self.scale1_bn = norm_layer(228*2)
        self.scale1_relu = nn.ReLU(inplace=True)

        self.scale2 = nn.Linear(228, 228)
        self.scale2_bn = norm_layer(228)
        self.scale2_relu = nn.ReLU(inplace=True)

        self.scale3 = nn.Linear(228, 150)
        self.scale3_bn = norm_layer(150)
        self.scale3_relu = nn.ReLU(inplace=True)

        self.scale4 = nn.Linear(228, 100)
        self.scale4_bn = norm_layer(100)
        self.scale4_relu = nn.ReLU(inplace=True)

        self.scale5 = nn.Linear(228, 80)
        self.scale5_bn = norm_layer(80)
        self.scale5_relu = nn.ReLU(inplace=True)

        self.scale6 = nn.Linear(228, 60)
        self.scale6_bn = norm_layer(60)
        self.scale6_relu = nn.ReLU(inplace=True)

        self.scale7 = nn.Linear(228, 40)
        self.scale7_bn = norm_layer(40)
        self.scale7_relu = nn.ReLU(inplace=True)

        self.scale8 = nn.Linear(228, 20)
        self.scale8_bn = norm_layer(20)
        self.scale8_relu = nn.ReLU(inplace=True)

        self.final_input_bn = norm_layer(228*2+228+150+100+80+60+40+20)
        self.final_input = nn.Linear(228*2+228+150+100+80+60+40+20, 256)

        self.final_input_relu = nn.ReLU(inplace=True)

        for m in self.modules():
            # if isinstance(m, nn.Conv2d):
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            # elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

        # # [128, 256, 128] => [128, 64, 32]
        # # TODO embd FC
        # self.embd_size = 32
        # self.embd_fc = nn.Sequential(
        #     # fc_1
        #     nn.Linear(in_features=128 * block.expansion, out_features=64),
        #     nn.BatchNorm1d(64),
        #     nn.LeakyReLU(),
        #     # fc_2
        #     nn.Linear(in_features=64, out_features=self.embd_size),
        #     nn.BatchNorm1d(self.embd_size),  # BatchNomal 保留
        #     # TODO embd 输出不要激活函数，效果提升 1 个百分点
        # )
        # pass

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):

        # orgin = x
        #
        # x = self.initial_fe1(x)
        # x = self.initial_bn1(x)
        # x = self.initial_relu1(x)
        # x = self.initial_fe2(x)
        # x = self.initial_bn2(x)
        # x = self.initial_relu2(x)

        # x = self.initial_fe3(x)
        # x = self.initial_bn3(x)
        # x = self.initial_relu3(x)

        s1 = self.scale1(x)
        # s1 = self.scale1_bn(s1)
        # s1 = self.scale1_relu(s1)

        s2 = self.scale2(x)
        # s2 = self.scale2_bn(s2)
        # s2 = self.scale2_relu(s2)

        s3 = self.scale3(x)
        # s3 = self.scale3_bn(s3)
        # s3 = self.scale3_relu(s3)

        s4 = self.scale4(x)
        # s4 = self.scale4_bn(s4)
        # s4 = self.scale4_relu(s4)

        s5 = self.scale5(x)
        # s5 = self.scale5_bn(s5)
        # s5 = self.scale5_relu(s5)

        s6 = self.scale6(x)
        # s6 = self.scale6_bn(s6)
        # s6 = self.scale6_relu(s6)

        s7 = self.scale7(x)
        # s7 = self.scale7_bn(s7)
        # s7 = self.scale7_relu(s7)

        s8 = self.scale8(x)
        # s8 = self.scale8_bn(s8)
        # s8 = self.scale8_relu(s8)

        concat = torch.cat([s1, s2, s3, s4, s5, s6, s7, s8], 1)

        concat = self.final_input_bn(concat)

        concat = concat.view(
            [int(concat.size(dim=0)), int(concat.size(dim=1)), 1, 1])
        concat = self.refine_initial(concat)
        concat = concat.view(
            [int(concat.size(dim=0)), int(concat.size(dim=1))])

        x = self.final_input(concat)
        x = self.final_input_relu(x)
        # x = self.upsample(x)
        # x = self.downs(x)

        # print(x.size())
        x = x.view([int(x.size(dim=0)), 256, 1])
        #
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x)

        x = self.layer1(x)
        b = x
        x = x.view(int(x.size(dim=0)),  int(
            x.size(dim=1))*int(x.size(dim=2)), 1, 1)
        x = self.refine1(x)
        x = x.view(int(b.size(dim=0)), int(b.size(dim=1)), int(b.size(dim=2)))

        x = self.layer2(x)

        a = x

        x = x.view(int(x.size(dim=0)),  int(
            x.size(dim=1))*int(x.size(dim=2)), 1, 1)
        x = self.refine2(x)
        x = x.view(int(a.size(dim=0)), int(a.size(dim=1)), int(a.size(dim=2)))

        x = self.layer3(x)

        c = x

        x = x.view(int(x.size(dim=0)),  int(
            x.size(dim=1))*int(x.size(dim=2)), 1, 1)
        x = self.refine3(x)
        x = x.view(int(c.size(dim=0)), int(c.size(dim=1)), int(c.size(dim=2)))

        x = self.layer4(x)

        d = x

        x = x.view(int(x.size(dim=0)), int(
            x.size(dim=1)) * int(x.size(dim=2)), 1, 1)
        x = self.refine4(x)
        x = x.view(int(d.size(dim=0)), int(d.size(dim=1)), int(d.size(dim=2)))

        # print(x.size())
        x = x.view(x.size(0), -1)

        embd = x

        # print(embd.size())

        pred1 = self.fc1(embd)
        pred1 = self.fc11(pred1)
        pred1 = self.soft1(pred1)

        pred2 = self.fc2(embd)
        pred2 = self.fc22(pred2)
        pred2 = self.soft2(pred2)

        pred3 = self.fc3(embd)
        pred3 = self.fc33(pred3)
        pred3 = self.soft3(pred3)

        pred4 = self.fc4(embd)
        pred4 = self.fc44(pred4)
        pred4 = self.soft4(pred4)

        pred5 = self.fc5(embd)
        pred5 = self.fc55(pred5)
        pred5 = self.soft5(pred5)

        pred6 = self.fc6(embd)
        pred6 = self.fc66(pred6)
        pred6 = self.soft6(pred6)

        pred7 = self.fc7(embd)
        pred7 = self.fc77(pred7)
        pred7 = self.soft7(pred7)

        pred8 = self.fc8(embd)
        pred8 = self.fc88(pred8)
        pred8 = self.soft8(pred8)

        pred9 = self.fc9(embd)
        pred9 = self.fc99(pred9)
        pred9 = self.soft9(pred9)

        pred10 = self.fc10(embd)
        pred10 = self.fc1010(pred10)
        pred10 = self.soft10(pred10)

        pred11 = self.fc111(embd)
        pred11 = self.fc1111(pred11)
        pred11 = self.soft11(pred11)

        pred12 = self.fc12(embd)
        pred12 = self.fc1212(pred12)
        pred12 = self.soft12(pred12)

        pred13 = self.fc13(embd)
        pred13 = self.fc1313(pred13)
        pred13 = self.soft13(pred13)

        # TODO
        # embd = self.embd_fc(embd)
        return pred1, pred2, pred3, pred4, pred5, pred6, pred7, pred8, pred9, pred10, pred11, pred12, pred13


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=61, in_ch=5, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            # norm_layer = nn.BatchNorm2d
            norm_layer = nn.BatchNorm1d
        self._norm_layer = norm_layer

        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        # self.inplanes = 64
        self.inplanes = 16
        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group
        # self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
        #                        bias=False)
        # TODO
        self.conv1 = nn.Conv1d(in_ch, self.inplanes, kernel_size=7, stride=1, padding=3,
                               bias=False)

        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        layer_channels = [64, 128, 256, 512]
        # layer_channels = [256, 512, 1024, 2048]
        # layer_channels = [32, 64, 128, 256]

        self.refine1 = ContextBlock(inplanes=3648*4, ratio=1. / 4., )
        self.refine2 = ContextBlock(inplanes=3712*4, ratio=1. / 4., )
        self.refine3 = ContextBlock(inplanes=3840*4, ratio=1. / 4., )
        self.refine4 = ContextBlock(inplanes=4096*4, ratio=1. / 4., )

        # self.refine1 = ContextBlock(inplanes=7296, ratio=1. / 4., )
        # self.refine2 = ContextBlock(inplanes=7424, ratio=1. / 4., )
        # self.refine3 = ContextBlock(inplanes=7680, ratio=1. / 4., )
        # self.refine4 = ContextBlock(inplanes=8192, ratio=1. / 4., )
        # layer_channels = [256, 512, 1024, 2048]
        self.layer1 = self._make_layer(block, layer_channels[0], layers[0])
        self.layer2 = self._make_layer(block, layer_channels[1], layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, layer_channels[2], layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, layer_channels[3], layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        # self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.fc1 = nn.Linear(4096*4, 256)
        # self.fc1soft1 = nn.Softmax(dim=1)
        self.fc11 = nn.Linear(256, 2)
        #

        self.fc2 = nn.Linear(4096*4, 256)
        # self.fc2soft1 = nn.Softmax(dim=1)
        self.fc22 = nn.Linear(256, 2)
        #

        self.soft1 = nn.Softmax(dim=1)
        self.soft2 = nn.Softmax(dim=1)

        for m in self.modules():
            # if isinstance(m, nn.Conv2d):
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            # elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

        # # [128, 256, 128] => [128, 64, 32]
        # # TODO embd FC
        # self.embd_size = 32
        # self.embd_fc = nn.Sequential(
        #     # fc_1
        #     nn.Linear(in_features=128 * block.expansion, out_features=64),
        #     nn.BatchNorm1d(64),
        #     nn.LeakyReLU(),
        #     # fc_2
        #     nn.Linear(in_features=64, out_features=self.embd_size),
        #     nn.BatchNorm1d(self.embd_size),  # BatchNomal 保留
        #     # TODO embd 输出不要激活函数，效果提升 1 个百分点
        # )
        # pass

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):

        # print(x.size())
        x = x.view([int(x.size(dim=0)), 4, 57])
        #
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x)

        x = self.layer1(x)
        b = x
        x = x.view(int(x.size(dim=0)),  int(
            x.size(dim=1))*int(x.size(dim=2)), 1, 1)
        x = self.refine1(x)
        x = x.view(int(b.size(dim=0)), int(b.size(dim=1)), int(b.size(dim=2)))

        x = self.layer2(x)

        a = x

        x = x.view(int(x.size(dim=0)),  int(
            x.size(dim=1))*int(x.size(dim=2)), 1, 1)
        x = self.refine2(x)
        x = x.view(int(a.size(dim=0)), int(a.size(dim=1)), int(a.size(dim=2)))

        x = self.layer3(x)

        c = x

        x = x.view(int(x.size(dim=0)),  int(
            x.size(dim=1))*int(x.size(dim=2)), 1, 1)
        x = self.refine3(x)
        x = x.view(int(c.size(dim=0)), int(c.size(dim=1)), int(c.size(dim=2)))

        x = self.layer4(x)

        d = x

        x = x.view(int(x.size(dim=0)), int(
            x.size(dim=1)) * int(x.size(dim=2)), 1, 1)
        x = self.refine4(x)
        x = x.view(int(d.size(dim=0)), int(d.size(dim=1)), int(d.size(dim=2)))

        x = x.view(x.size(0), -1)

        embd = x

        pred1 = self.fc1(embd)

        pred1 = self.fc11(pred1)
        pred1 = self.soft1(pred1)
        # pred1 = self.cosdis1(embd)

        pred2 = self.fc2(embd)
        pred2 = self.fc22(pred2)
        # print(pred2.size())
        pred2 = self.soft2(pred2)
        # print(pred2.size())
        # pred2 = self.cosdis2(embd)

        # TODO
        # embd = self.embd_fc(embd)
        return pred1, pred2


def _resnet(arch, block, layers, progress, pretrained=False, **kwargs):
    model = ResNet(block, layers, **kwargs)
    return model


def get_model(M, L, C,num_classes):
    model_18 = ResNet_d(
        block=BasicBlock_d, layers=[1, 1, 1, 1],
        width_per_group=L,
        in_ch=1,num_classes=num_classes
    )
    return model_18

def Net(conf:Config,device):
    if conf.dataset.axis is None:
        return [get_model(4, 61, 3,conf.dataset.num_classes).to(device) for _ in range(3)]
    return get_model(4, 61, 3,conf.dataset.num_classes).to(device)
#
# def get_model(M, L, C):
#     model_18 = ResNet_c(
#         block=BasicBlock_c, layers=[1, 1, 1, 1], num_classes=C,
#         width_per_group=L,
#         in_ch=256
#     )
#     return model_18

# def get_model(M, L, C):
#     model_18 = ResNet(
#         block=Bottleneck, layers=[3, 4, 6, 3], num_classes=C,
#         width_per_group=L,
#         in_ch=4
#     )
#     return model_18

#
# def get_model(M, L, C):
#     model_18 = Custom()
#     return model_18


def get_model_res9(M=5, L=61, C=3):
    model_9 = ResNet(
        block=Bottleneck,
        layers=[1, 1, 1, 1],
        num_classes=C,
        width_per_group=L,
        in_ch=M
    )
    temp_18 = ResNet(
        block=BasicBlock,
        layers=[2, 2, 2, 2],
        num_classes=C,
        width_per_group=L,
        in_ch=M
    )
    model_18 = ResNet(
        block=Bottleneck, layers=[2, 2, 2, 2], num_classes=C,
        width_per_group=L,
        in_ch=M
    )
    temp_9 = ResNet(
        block=BasicBlock,
        layers=[1, 1, 1, 1],
        num_classes=C,
        width_per_group=L,
        in_ch=M
    )

    from torchvision import models
    temp = models.resnet18()
    return model_9





if __name__ == '__main__':
    get_model_res9()
    pass
