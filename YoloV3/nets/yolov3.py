from collections import OrderedDict

import torch
import torch.nn as nn

from YoloV3.nets.darknet import darknet53


def conv2d(filter_in, filter_out, kernel_size):
    """
    创建一个包含卷积层、批量归一化层和激活层的序列。

    参数:
        filter_in (int): 输入通道数。
        filter_out (int): 输出通道数。
        kernel_size (int): 卷积核大小。

    返回:
        nn.Sequential: 包含卷积层、批量归一化层和激活层的序列。
    """
    # 计算填充大小
    pad = (kernel_size - 1) // 2 if kernel_size else 0

    # 创建包含卷积层、批量归一化层和激活层的序列
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=1, padding=pad, bias=False)),
        ("bn", nn.BatchNorm2d(filter_out)),
        ("relu", nn.LeakyReLU(0.1)),
    ]))


def make_last_layers(filters_list, in_filters, out_filter):
    """
    创建最后一组卷积层序列，用于提取特征并输出。

    参数:
        filters_list (list): 中间卷积层的通道数列表。
        in_filters (int): 输入通道数。
        out_filter (int): 最终输出通道数。

    返回:
        nn.ModuleList: 包含多个卷积层的模块列表。
    """
    # 创建包含多个卷积层的模块列表
    m = nn.ModuleList([
        conv2d(in_filters, filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        nn.Conv2d(filters_list[1], out_filter, kernel_size=1,
                  stride=1, padding=0, bias=True)
    ])
    return m


class YoloBody(nn.Module):
    def __init__(self, anchor, num_classes):
        """
        YOLOv3模型主体初始化。

        参数:
            anchor (list): 锚框尺寸列表，按特征层级排列。
            num_classes (int): 目标类别总数。
        """
        super(YoloBody, self).__init__()

        # 使用Darknet-53作为骨干网络
        self.backbone = darknet53(None)

        # 获取骨干网络输出特征层的通道数  self.backbone.layers_out_filters = [64, 128, 256, 512, 1024]
        out_filters = self.backbone.layers_out_filters

        # 计算第一层级输出通道数，包括框的位置、大小、对象得分和各类别概率
        final_out_filter0 = len(anchor[0]) * (5 + num_classes)
        # 构建第一层级的最后一部分网络结构
        self.last_layer0 = make_last_layers([512, 1024], out_filters[-1], final_out_filter0)

        # 计算第二层级输出通道数
        final_out_filter1 = len(anchor[1]) * (5 + num_classes)
        # 构建从第一层级到第二层级的过渡层和第二层级的最后一部分网络结构
        self.last_layer1_conv = conv2d(512, 256, 1)
        self.last_layer1_upsample = nn.Upsample(scale_factor=2, mode='nearest')  # 上采样层
        self.last_layer1 = make_last_layers([256, 512], out_filters[-2] + 256, final_out_filter1)

        # 计算第三层级输出通道数
        final_out_filter2 = len(anchor[2]) * (5 + num_classes)
        # 构建从第二层级到第三层级的过渡层和第三层级的最后一部分网络结构
        self.last_layer2_conv = conv2d(256, 128, 1)
        self.last_layer2_upsample = nn.Upsample(scale_factor=2, mode='nearest')  # 上采样层
        self.last_layer2 = make_last_layers([128, 256], out_filters[-3] + 128, final_out_filter2)

    def forward(self, x):
        """
        YOLOv3模型前向传播过程。

        参数:
            x (torch.Tensor): 输入图像，形状通常为[N, C, H, W]。

        返回:
            tuple(torch.Tensor): 各层级输出特征图。
        """
        # 定义一个内部辅助函数用于处理分支操作
        def _branch(last_layer, layer_in):
            for i, e in enumerate(last_layer):
                layer_in = e(layer_in)
                # 在第4个元素处分支保存输出
                if i == 4:
                    out_branch = layer_in
            return layer_in, out_branch

        # 获得骨干网络输出特征图
        x2, x1, x0 = self.backbone(x)

        # 第一层级的处理
        out0, out0_branch = _branch(self.last_layer0, x0)

        # 构造第二层级输入：上采样+拼接
        x1_in = self.last_layer1_conv(out0_branch)
        x1_in = self.last_layer1_upsample(x1_in)
        x1_in = torch.cat([x1_in, x1], 1)  # 拼接特征图

        # 第二层级的处理
        out1, out1_branch = _branch(self.last_layer1, x1_in)

        # 构造第三层级输入：上采样+拼接
        x2_in = self.last_layer2_conv(out1_branch)
        x2_in = self.last_layer2_upsample(x2_in)
        x2_in = torch.cat([x2_in, x2], 1)  # 拼接特征图

        # 第三层级的处理
        out2, _ = _branch(self.last_layer2, x2_in)

        # 返回各层级输出
        return out0, out1, out2