# -*- coding: <encoding name> -*-
"""
PSNet model
"""
from __future__ import print_function, division
import torch.nn as nn
import torch
from torch.utils import model_zoo
import numpy as np
import torch.nn.functional as F


################################################################################
# PSNet
################################################################################
class PSNet(nn.Module):
    def __init__(self):
        super(PSNet, self).__init__()
        self.vgg = VGG()
        self.dmp = BackEnd()

        self._load_vgg()

    def forward(self, input):
        input = self.vgg(input)
        dmp_out = self.dmp(*input)

        return dmp_out

    def _load_vgg(self):
        state_dict = model_zoo.load_url('https://download.pytorch.org/models/vgg16_bn-6c64b313.pth')
        old_name = [0, 1, 3, 4, 7, 8, 10, 11, 14, 15, 17, 18, 20, 21, 24, 25, 27, 28, 30, 31, 34, 35, 37, 38, 40, 41]
        new_name = ['1_1', '1_2', '2_1', '2_2', '3_1', '3_2', '3_3', '4_1', '4_2', '4_3', '5_1', '5_2', '5_3']
        new_dict = {}
        for i in range(10):
            new_dict['conv' + new_name[i] + '.conv.weight'] = \
                state_dict['features.' + str(old_name[2 * i]) + '.weight']
            new_dict['conv' + new_name[i] + '.conv.bias'] = \
                state_dict['features.' + str(old_name[2 * i]) + '.bias']
            new_dict['conv' + new_name[i] + '.bn.weight'] = \
                state_dict['features.' + str(old_name[2 * i + 1]) + '.weight']
            new_dict['conv' + new_name[i] + '.bn.bias'] = \
                state_dict['features.' + str(old_name[2 * i + 1]) + '.bias']
            new_dict['conv' + new_name[i] + '.bn.running_mean'] = \
                state_dict['features.' + str(old_name[2 * i + 1]) + '.running_mean']
            new_dict['conv' + new_name[i] + '.bn.running_var'] = \
                state_dict['features.' + str(old_name[2 * i + 1]) + '.running_var']

        self.vgg.load_state_dict(new_dict)


class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1_1 = BaseConv(3, 64, 3, 1, 1, activation=nn.ReLU(), use_bn=True)
        self.conv1_2 = BaseConv(64, 64, 3, 1, 1, activation=nn.ReLU(), use_bn=True)
        self.conv2_1 = BaseConv(64, 128, 3, 1, 1, activation=nn.ReLU(), use_bn=True)
        self.conv2_2 = BaseConv(128, 128, 3, 1, 1, activation=nn.ReLU(), use_bn=True)
        self.conv3_1 = BaseConv(128, 256, 3, 1, 1, activation=nn.ReLU(), use_bn=True)
        self.conv3_2 = BaseConv(256, 256, 3, 1, 1, activation=nn.ReLU(), use_bn=True)
        self.conv3_3 = BaseConv(256, 256, 3, 1, 1, activation=nn.ReLU(), use_bn=True)
        self.conv4_1 = BaseConv(256, 512, 3, 1, 1, activation=nn.ReLU(), use_bn=True)
        self.conv4_2 = BaseConv(512, 512, 3, 1, 1, activation=nn.ReLU(), use_bn=True)
        self.conv4_3 = BaseConv(512, 512, 3, 1, 1, activation=nn.ReLU(), use_bn=True)

    def forward(self, input):
        input = self.conv1_1(input)
        conv1_2 = self.conv1_2(input)

        input = self.pool(conv1_2)
        input = self.conv2_1(input)
        conv2_2 = self.conv2_2(input)

        input = self.pool(conv2_2)
        input = self.conv3_1(input)
        input = self.conv3_2(input)
        conv3_3 = self.conv3_3(input)

        input = self.pool(conv3_3)
        conv4_1 = self.conv4_1(input)
        input = self.conv4_2(conv4_1)
        conv4_3 = self.conv4_3(input)

        return conv1_2, conv3_3, conv4_1, conv4_3

class Decoder(nn.Module):
    def __init__(self, C3_size, C4_size, C5_size, feature_size=256):
        super(Decoder, self).__init__()

        # upsample C5 to get P5 from the FPN paper
        self.P5_1 = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_upsampled = nn.Upsample(scale_factor=1, mode='nearest')
        self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P5 elementwise to C4
        self.P4_1 = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P4 elementwise to C3
        self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)


    def forward(self, inputs):
        C3, C4, C5 = inputs

        P5_x = self.P5_1(C5)
        P5_upsampled_x = self.P5_upsampled(P5_x)
        P5_x = self.P5_2(P5_x)

        P4_x = self.P4_1(C4)
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = self.P4_upsampled(P4_x)
        P4_x = self.P4_2(P4_x)

        P3_x = self.P3_1(C3)
        P3_x = P3_x + P4_upsampled_x
        P3_x = self.P3_2(P3_x)

        return [P3_x, P4_x, P5_x]

class BackEnd(nn.Module):
    def __init__(self):
        super(BackEnd, self).__init__()

        self.att = ATT(512)
        self.se = SELayer(512)
        self.se1 = SELayer(256)
        self.se2 = SELayer(128)
        self.dense1 = DenseModule(512)
        self.dense2 = DenseModule(512)
        self.dense3 = DenseModule(512)
        self.dense4 = DenseModule(64)
        self.dense5 = DenseModule(128)
        self.dense6 = DenseModule(256)
        self.conv11 = BaseConv(1536, 1024, 3, 1, 1, activation=nn.ReLU(), use_bn=True)
        self.conv12 = BaseConv(1024, 512, 3, 1, 1, activation=nn.ReLU(), use_bn=True)
        self.conv1 = BaseConv(512, 256, 3, 1, 1, activation=nn.ReLU(), use_bn=True)
        self.conv2 = BaseConv(256, 128, 3, 1, 1, activation=nn.ReLU(), use_bn=True)
        self.conv3 = BaseConv(128, 64, 3, 1, 1, activation=nn.ReLU(), use_bn=True)
        self.conv4 = BaseConv(64, 1, 1, 1, activation=None, use_bn=False)
        self.ssh = SSH(512, 512)
        self.AGAM = ChannelAttention(512)
        self.fpn = Decoder(256, 512, 512)

    def forward(self, *input):
        conv1_2, conv3_3, conv4_1, conv4_3 = input
        # x = self.conv12(x)
        x = self.att(conv4_3)
        # x = self.AGAM(conv4_3)
        # conv4_3 = self.ssh(conv4_3)
        input1, attention_map_1 = self.dense1(conv4_3)
        # attention_map_1 *= x
        # input1 = input1 + conv4_3
        input, attention_map_2 = self.dense2(input1)
        # input2 = input2 + input1
        # attention_map_2 *= x
        input, attention_map_3 = self.dense3(input)
        # input = input + input2
        # input, attention_map_4 = self.dense3(input)


        # input = self.se(input)
        # input = self.ssh(input)
        # input *= x

        input = self.conv1(input)
        # input = self.se1(input)
        input = self.conv2(input)
        # input = self.se2(input)
        input = self.conv3(input)
        input = self.conv4(input)
        # input = torch.cat((input, conv3_3), dim=1)


        return input, attention_map_1, attention_map_2, attention_map_3

class SSH(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(SSH, self).__init__()
        assert out_channel % 4 == 0
        leaky = 0
        if (out_channel <= 64):
            leaky = 0.1
        self.conv3X3 = conv_bn_no_relu(in_channel, out_channel//2, stride=1)

        self.conv5X5_1 = conv_bn(in_channel, out_channel//4, stride=1, leaky = leaky)
        self.conv5X5_2 = conv_bn_no_relu(out_channel//4, out_channel//4, stride=1)

        self.conv7X7_2 = conv_bn(out_channel//4, out_channel//4, stride=1, leaky = leaky)
        self.conv7x7_3 = conv_bn_no_relu(out_channel//4, out_channel//4, stride=1)

    def forward(self, input):
        conv3X3 = self.conv3X3(input)

        conv5X5_1 = self.conv5X5_1(input)
        conv5X5 = self.conv5X5_2(conv5X5_1)

        conv7X7_2 = self.conv7X7_2(conv5X5_1)
        conv7X7 = self.conv7x7_3(conv7X7_2)

        out = torch.cat([conv3X3, conv5X5, conv7X7], dim=1)
        out = F.relu(out)
        return out
def conv_bn_no_relu(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
    )
def conv_bn(inp, oup, stride = 1, leaky = 0):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True)
    )
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv1 = BaseConv(in_planes, round(in_planes // ratio), 1, 1, activation=nn.ReLU(), use_bn=False)
        self.conv2 = BaseConv(round(in_planes // ratio), round(in_planes // ratio), 1, 1, activation=nn.Sigmoid(), use_bn=False)

        self.conv3 = BaseConv(round(in_planes // ratio), in_planes, 1, 1, activation=nn.Sigmoid(), use_bn=False)
    def forward(self, input):
        # out = self.max_pool(input)
        out = self.conv1(input)
        # out = self.conv2(out)
        out = self.conv3(out)

        return out


class DenseModule(nn.Module):
    def __init__(self, in_channels):
        super(DenseModule, self).__init__()

        self.conv3x3 = nn.Sequential(
            BaseConv(in_channels, in_channels // 4, 1, 1, activation=nn.ReLU(), use_bn=True),
            BaseConv(in_channels // 4, in_channels // 4, 3, 1, 1, activation=nn.ReLU(), use_bn=True))
        self.conv5x5 = nn.Sequential(
            BaseConv(in_channels, in_channels // 4, 1, 1, activation=nn.ReLU(), use_bn=True),
            BaseConv(in_channels // 4, in_channels // 4, 3, 1, 2, 2, activation=nn.ReLU(), use_bn=True))
        self.conv7x7 = nn.Sequential(
            BaseConv(in_channels, in_channels // 4, 1, 1, activation=nn.ReLU(), use_bn=True),
            BaseConv(in_channels // 4, in_channels // 4, 3, 1, 3, 3, activation=nn.ReLU(), use_bn=True))
        self.conv9x9 = nn.Sequential(
            BaseConv(in_channels, in_channels // 4, 1, 1, activation=nn.ReLU(), use_bn=True),
            BaseConv(in_channels // 4, in_channels // 4, 3, 1, 4, 4, activation=nn.ReLU(), use_bn=True))
        self.conv11x11 = nn.Sequential(
            BaseConv(in_channels, in_channels // 4, 1, 1, activation=nn.ReLU(), use_bn=True),
            BaseConv(in_channels // 4, in_channels // 4, 3, 1, 5, 5, activation=nn.ReLU(), use_bn=True))
        self.conv13x13 = nn.Sequential(
            BaseConv(in_channels, in_channels // 4, 1, 1, activation=nn.ReLU(), use_bn=True),
            BaseConv(in_channels // 4, in_channels // 4, 3, 1, 6, 6, activation=nn.ReLU(), use_bn=True))
        self.up = BaseConv(in_channels,in_channels // 4, 1, 1, activation=nn.ReLU(), use_bn=True)
        self.conv1 = BaseConv(in_channels // 2, in_channels // 4, 3, 1, 2, 2, activation=nn.ReLU(), use_bn=True)
        self.conv2 = BaseConv(in_channels // 2, in_channels // 4, 3, 1, 2, 2,  activation=nn.ReLU(), use_bn=True)
        self.conv3 = BaseConv(in_channels // 2, in_channels // 4, 3, 1, 2, 2,  activation=nn.ReLU(), use_bn=True)

        self.att = ChannelAttention(in_channels)
        self.ca = CoordAttention(in_channels,in_channels)
        self.conv = BaseConv(in_channels//4*5, in_channels, 3, 1, 1, activation=nn.ReLU(), use_bn=True)
        self.se = SELayer(512)
        self.se1 = SELayer(4)
        self.ssh = SSH(512, 512)
        self.ssh1 = SSH(128, 128)
    def forward(self, input):
        conv3x3 = self.conv3x3(input)
        conv5x5 = self.conv5x5(input)
        conv7x7 = self.conv7x7(input)
        conv9x9 = self.conv9x9(input)
        # conv11x11 = self.conv11x11(input)
        # conv13x13 = self.conv13x13(input)

        conv5x5 = self.conv1(torch.cat((conv3x3, conv5x5), dim=1))
        conv7x7 = self.conv2(torch.cat((conv5x5, conv7x7), dim=1))
        conv9x9 = self.conv3(torch.cat((conv7x7, conv9x9), dim=1))
        # conv11x11 = self.conv3(torch.cat((conv9x9, conv11x11), dim=1))
        # conv13x13 = self.conv3(torch.cat((conv11x11, conv13x13), dim=1))

        # conv5x5 = self.ssh1(conv5x5)
        # conv7x7 = self.ssh1(conv7x7)
        # conv9x9 = self.ssh1(conv9x9)
        att = self.att(input)
        up = self.up(input)
        out = self.conv(torch.cat((up, conv3x3, conv5x5, conv7x7, conv9x9), dim=1))
        # out = self.conv(torch.cat((up, conv3x3, conv5x5, conv7x7, conv9x9, conv11x11),dim=1))
        # out = up
        out = self.se(out)
        # out = self.ca(out)
        out = self.ssh(out)
        out = out*att
        attention_map = torch.cat((torch.mean(up, dim=1, keepdim=True),
                                   torch.mean(conv3x3, dim=1, keepdim=True),
                                   torch.mean(conv5x5, dim=1, keepdim=True),
                                   torch.mean(conv7x7, dim=1, keepdim=True),
                                   torch.mean(conv9x9, dim=1, keepdim=True)), dim=1)
        # attention_map = self.se1(attention_map)
        # attention_map = self.ssh1(attention_map)
        return out , attention_map


class BaseConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride=1, padding=0, dilation=1, activation=None,
                 use_bn=False):
        super(BaseConv, self).__init__()
        self.use_bn = use_bn
        self.activation = activation
        self.conv = nn.Conv2d(in_channels, out_channels, kernel, stride, padding, dilation)
        self.conv.weight.data.normal_(0, 0.01)
        self.conv.bias.data.zero_()
        self.bn = nn.BatchNorm2d(out_channels)
        self.bn.weight.data.fill_(1)
        self.bn.bias.data.zero_()

    def forward(self, input):
        input = self.conv(input)
        if self.use_bn:
            input = self.bn(input)
        if self.activation:
            input = self.activation(input)

        return input



class ATT(nn.Module):

    def __init__(self, channels, reduction=16):
        super(ATT, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1,
                             padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1,
                             padding=0)
        self.sigmoid_channel = nn.Sigmoid()
        #self.conv_after_concat = nn.Conv2d(2, 1, kernel_size = 3, stride=1, padding = 1)



    def forward(self, x):
        module_input = x
        avg = self.avg_pool(x)
        mx = self.max_pool(x)
        avg = self.fc1(avg)
        mx = self.fc1(mx)
        avg = self.relu(avg)
        mx = self.relu(mx)
        avg = self.fc2(avg)
        mx = self.fc2(mx)
        x = avg   + mx
        x = self.sigmoid_channel(x)
        return x
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

import torch
import torch.nn as nn


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAttention(nn.Module):

    def __init__(self, in_channels, out_channels, reduction=32):
        super(CoordAttention, self).__init__()
        self.pool_w, self.pool_h = nn.AdaptiveAvgPool2d((1, None)), nn.AdaptiveAvgPool2d((None, 1))
        temp_c = max(8, in_channels // reduction)
        self.conv1 = nn.Conv2d(in_channels, temp_c, kernel_size=1, stride=1, padding=0)

        self.bn1 = nn.BatchNorm2d(temp_c)
        self.act1 = h_swish()

        self.conv2 = nn.Conv2d(temp_c, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(temp_c, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        short = x
        n, c, H, W = x.shape
        x_h, x_w = self.pool_h(x), self.pool_w(x).permute(0, 1, 3, 2)
        x_cat = torch.cat([x_h, x_w], dim=2)
        out = self.act1(self.bn1(self.conv1(x_cat)))
        x_h, x_w = torch.split(out, [H, W], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        out_h = torch.sigmoid(self.conv2(x_h))
        out_w = torch.sigmoid(self.conv3(x_w))
        return short * out_w * out_h
