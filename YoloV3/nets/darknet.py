import math
from collections import OrderedDict

import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes):
        """
        初始化基本残差块。

        参数:
            inplanes (int): 输入特征图的通道数。
            planes (tuple): 每个卷积层的输出通道数。
        """
        super(BasicBlock, self).__init__()

        # 第一个卷积层：输入通道为inplanes，输出通道为planes[0]，卷积核大小为1x1，步长为1，填充为0，无偏置
        self.conv1 = nn.Conv2d(inplanes, planes[0], kernel_size=1,
                               stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes[0])  # 批量归一化层
        self.relu1 = nn.LeakyReLU(0.1)  # LeakyReLU 激活层

        # 第二个卷积层：输入通道为planes[0]，输出通道为planes[1]，卷积核大小为3x3，步长为1，填充为1，无偏置
        self.conv2 = nn.Conv2d(planes[0], planes[1], kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes[1])  # 批量归一化层
        self.relu2 = nn.LeakyReLU(0.1)  # LeakyReLU 激活层

    def forward(self, x):
        """
        前向传播过程。

        参数:
            x (torch.Tensor): 输入张量。

        返回:
            torch.Tensor: 输出张量。
        """
        # 保存残差连接的输入
        residual = x

        # 第一个卷积层
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        # 第二个卷积层
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        # 残差连接
        out += residual

        return out


class DarkNet(nn.Module):
    def __init__(self, layers):
        """
        初始化 DarkNet 模型。

        参数:
            layers (list): 各个残差块的数量列表。
        """
        super(DarkNet, self).__init__()

        # 初始化输入平面数为32
        self.inplanes = 32

        # 416x416x3 -> 416x416x32
        # 卷积层：输入通道为3，输出通道为32，卷积核大小为3x3，步长为1，填充为1，无偏置
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)  # 批量归一化层
        self.relu1 = nn.LeakyReLU(0.1)  # LeakyReLU 激活层

        # 416x416x32 -> 208x208x64
        # 构建第一个残差块层
        self.layer1 = self._make_layer([32, 64], layers[0])
        # 208x208x64 -> 104x104x128
        # 构建第二个残差块层
        self.layer2 = self._make_layer([64, 128], layers[1])
        # 104x104x128 -> 52x52x256
        # 构建第三个残差块层
        self.layer3 = self._make_layer([128, 256], layers[2])
        # 52x52x256 -> 26x26x512
        # 构建第四个残差块层
        self.layer4 = self._make_layer([256, 512], layers[3])
        # 26x26x512 -> 13x13x1024
        # 构建第五个残差块层
        self.layer5 = self._make_layer([512, 1024], layers[4])

        # 记录各层输出的特征图通道数
        self.layers_out_filters = [64, 128, 256, 512, 1024]

        # 初始化权重和偏置
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 计算卷积层参数数量
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # 初始化权重，采用正态分布
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                # 初始化批量归一化层的权重和偏置
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, planes, blocks):
        """
        构建一个包含多个残差块的层。

        参数:
            planes (tuple): 每个残差块的通道数配置。
            blocks (int): 该层中包含的残差块数量。

        返回:
            nn.Sequential: 包含所有层的序列化模块。
        """

        # 初始化一个空列表来存储层
        layers = []

        # 添加下采样卷积层
        layers.append(("ds_conv", nn.Conv2d(self.inplanes, planes[1], kernel_size=3,
                                            stride=2, padding=1, bias=False)))  # 下采样卷积层
        layers.append(("ds_bn", nn.BatchNorm2d(planes[1])))  # 批量归一化层
        layers.append(("ds_relu", nn.LeakyReLU(0.1)))  # LeakyReLU激活层

        # 更新输入平面数
        self.inplanes = planes[1]

        # 循环添加多个残差块
        for i in range(0, blocks):
            # 为每个残差块命名，并将其添加到layers列表中
            layers.append(("residual_{}".format(i), BasicBlock(self.inplanes, planes)))

        # 将layers列表转换为有序字典，并创建一个nn.Sequential对象
        return nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        out3 = self.layer3(x)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)

        return out3, out4, out5


def darknet53(pretrained, **kwargs):
    model = DarkNet([1, 2, 8, 8, 4])
    if pretrained:
        if isinstance(pretrained, str):
            model.load_state_dict(torch.load(pretrained))
        else:
            raise Exception("darknet request a pretrained path. got [{}]".format(pretrained))
    return model
