import torch
import numpy as np
import warnings
import warnings
import torchvision.transforms.functional as TTF
from torchvision.transforms.functional import InterpolationMode
from collections.abc import Sequence
from typing import Tuple, List, Optional
import numbers
from torch import Tensor
import torchvision.transforms as transforms


class myRandomAffine(torch.nn.Module):

    def __init__(  # zy：添加translate的反向变化inverse_translate
            self, degrees, translate=None, inverse_translate=None, scale=None, inv_scale=None, shear=None,
            interpolation=InterpolationMode.NEAREST, fill=0,
            fillcolor=None, resample=None
    ):
        """
        初始化一个图像变换对象

        参数:
            degrees (float 或 tuple): 旋转角度。
            translate (tuple, 可选): 平移距离，格式为 (x, y)。
            inverse_translate (tuple, 可选): translate的反向平移距离。zy添加
            scale (tuple, 可选): 缩放比例，格式为 (width, height)。
            inv_scale (tuple, 可选): scale的反向缩放比例。zy添加
            shear (float 或 tuple, 可选): 错切角度。
            interpolation (InterpolationMode, 可选): 插值模式，默认为 InterpolationMode.NEAREST。
            fill (int 或 tuple, 可选): 用于填充新像素的颜色值或颜色序列，默认为 0。
            fillcolor (int 或 tuple, 可选): 已弃用，等效于 fill。
            resample (int, 可选): 重采样模式，默认为 None，表示使用 interpolation 的值。

        返回:
            None
        """

        super().__init__()

        # 检查并更新 fill 值
        if fillcolor is not None:
            warnings.warn(
                "Argument fillcolor is deprecated and will be removed since v0.10.0. Please, use fill instead"
            )
            fill = fillcolor

        # 设置旋转角度
        self.degrees = _setup_angle(degrees, name="degrees", req_sizes=(2,))

        # 设置平移
        if translate is not None:
            _check_sequence_input(translate, "translate", req_sizes=(2,))
            for t in translate:
                if not (0.0 <= t <= 1.0):
                    raise ValueError("translation values should be between 0 and 1")
        self.translate = translate

        # 添加反向平移的设置
        if inverse_translate is not None:
            _check_sequence_input(inverse_translate, "inverse_translate", req_sizes=(2,))
        self.inverse_translate = inverse_translate

        # 设置缩放
        if scale is not None:
            _check_sequence_input(scale, "scale", req_sizes=(2,))
            for s in scale:
                if s <= 0:
                    raise ValueError("scale values should be positive")
        self.scale = scale

        # 添加反向缩放的设置
        if inv_scale is not None:  # 反向缩放
            _check_sequence_input(inv_scale, "inv_scale", req_sizes=(2,))
        self.inv_scale = inv_scale

        # 设置错切
        if shear is not None:
            self.shear = _setup_angle(shear, name="shear", req_sizes=(2, 4))
        else:
            self.shear = shear

        # 设置插值模式
        self.interpolation = interpolation

        # 设置重采样模式
        self.resample = interpolation if resample is None else resample

        # 设置填充颜色或值
        if fill is None:
            fill = 0
        elif not isinstance(fill, (Sequence, numbers.Number)):
            raise TypeError("Fill should be either a sequence or a number.")

        self.fillcolor = self.fill = fill

    @staticmethod
    def get_params(
            degrees: List[float],
            translate: Optional[List[float]],
            inverse_translate: Optional[List[float]],  # zy添加
            scale_ranges: Optional[List[float]],
            inv_scale_ranges: Optional[List[float]],  # zy添加
            shears: Optional[List[float]],
            img_size: List[int]
    ) -> Tuple[float, Tuple[int, int], float, Tuple[float, float]]:
        """
        获取仿射变换的参数

        参数:
            degrees (List[float]): 角度范围。
            translate (Optional[List[float]]): 平移范围，格式为 [x, y]。
            inverse_translate (Optional[List[float]]): inverse_translate的反向平移范围。zy添加
            scale_ranges (Optional[List[float]]): 缩放范围，格式为 [min, max]。
            inv_scale_ranges (Optional[List[float]]): inv_scale_ranges的反向缩放范围。zy添加
            shears (Optional[List[float]]): 错切范围，格式为 [x, y]。
            img_size (List[int]): 图像尺寸，格式为 [width, height]。

        返回:
            一个元组，包含角度、平移、缩放和错切的值

        """

        # 随机生成一个角度值
        angle = float(torch.empty(1).uniform_(float(degrees[0]), float(degrees[1])).item())

        # 如果提供了平移范围
        if translate is not None:
            # 计算最大平移距离
            max_dx = float(translate[0] * img_size[0])
            max_dy = float(translate[1] * img_size[1])
            # 随机生成平移距离
            tx = int(round(torch.empty(1).uniform_(-max_dx, max_dx).item()))
            ty = int(round(torch.empty(1).uniform_(-max_dy, max_dy).item()))
            # 平移量
            translations = (tx, ty)
        elif inverse_translate is not None:  # zy添加反变化
            # 如果提供了反向平移量，则直接使用
            translations = inverse_translate
        else:
            # 默认情况下，平移量为0
            translations = (0, 0)

        # 如果提供了缩放范围
        if scale_ranges is not None:
            # 随机生成缩放比例
            scale = float(torch.empty(1).uniform_(scale_ranges[0], scale_ranges[1]).item())
        elif inv_scale_ranges is not None:  # zy添加反变化
            # 如果提供了反向缩放比例，则直接使用
            scale = inv_scale_ranges
        else:
            # 默认情况下，缩放比例为1.0
            scale = 1.0

        # 初始化错切值
        shear_x = shear_y = 0.0
        # 如果提供了错切范围
        if shears is not None:
            # 随机生成错切值x
            shear_x = float(torch.empty(1).uniform_(shears[0], shears[1]).item())
            # 如果提供了两个错切值，生成错切值y
            if len(shears) == 4:
                shear_y = float(torch.empty(1).uniform_(shears[2], shears[3]).item())

        # 错切量
        shear = (shear_x, shear_y)

        # 返回仿射变换的参数
        return angle, translations, scale, shear

    def forward(self, img):

        fill = self.fill
        # 如果输入的图像是一个张量
        if isinstance(img, Tensor):
            # 如果填充值是一个整数或浮点数，则创建一个包含该数字的列表，乘以图像的通道数
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * TTF._get_image_num_channels(img)
            # 如果填充值是一个列表，则创建一个包含每个值的浮点数列表
            else:
                fill = [float(f) for f in fill]

        # 获取图像的大小
        img_size = TTF._get_image_size(img)

        # 获取仿射变换的参数
        ret = self.get_params(self.degrees, self.translate, self.inverse_translate, self.scale, self.inv_scale,
                              self.shear, img_size)
        # zy添加: 将返回的元组拆分为单独的变量
        angle, translations, scale, shear = ret
        # 返回仿射变换后的图像，以及平移量和缩放系数
        return TTF.affine(img, *ret, interpolation=self.interpolation, fill=fill), translations, scale

    def __repr__(self):
        # 初始化字符串模板
        s = '{name}(degrees={degrees}'

        # 如果 translate 属性不为空，则添加到字符串中
        if self.translate is not None:
            s += ', translate={translate}'

        # 如果 scale 属性不为空，则添加到字符串中
        if self.scale is not None:
            s += ', scale={scale}'

        # 如果 shear 属性不为空，则添加到字符串中
        if self.shear is not None:
            s += ', shear={shear}'

        # 如果 interpolation 属性不是默认的 NEAREST，则添加到字符串中
        if self.interpolation != InterpolationMode.NEAREST:
            s += ', interpolation={interpolation}'

        # 如果 fill 属性不是默认的 0，则添加到字符串中
        if self.fill != 0:
            s += ', fill={fill}'

        # 结束字符串模板
        s += ')'

        # 将类的属性转换为字典
        d = dict(self.__dict__)

        # 将 interpolation 属性转换为其值
        d['interpolation'] = self.interpolation.value

        # 格式化字符串并返回
        return s.format(name=self.__class__.__name__, **d)


def _setup_angle(x, name, req_sizes=(2,)):
    # 检查变量 x 是否为数字类型
    if isinstance(x, numbers.Number):
        # 如果 x 是一个数字
        if x < 0:
            # 如果 x 是负数，抛出 ValueError
            raise ValueError("If {} is a single number, it must be positive.".format(name))
        # 如果 x 是正数，将其转换为一个范围 [-x, x]
        x = [-x, x]
    else:
        # 如果 x 不是数字类型，调用 _check_sequence_input 进行进一步验证
        _check_sequence_input(x, name, req_sizes)

    # 将 x 中的所有元素转换为浮点数
    return [float(d) for d in x]


def _check_sequence_input(x, name, req_sizes):
    # 定义一个变量 msg，用于表示请求的序列长度
    # 如果 req_sizes 只有一个元素，则直接使用该元素
    # 否则，将所有元素用 "or" 连接起来形成字符串
    msg = req_sizes[0] if len(req_sizes) < 2 else " or ".join([str(s) for s in req_sizes])

    # 检查变量 x 是否为序列类型
    if not isinstance(x, Sequence):
        # 如果 x 不是序列类型，则抛出 TypeError
        raise TypeError("{} should be a sequence of length {}.".format(name, msg))

    # 检查变量 x 的长度是否符合要求
    if len(x) not in req_sizes:
        # 如果 x 的长度不符合要求，则抛出 ValueError
        raise ValueError("{} should be sequence of length {}.".format(name, msg))
