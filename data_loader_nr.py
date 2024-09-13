from utils import get_params
import torch.nn.functional as F
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    # OpenGLPerspectiveCameras,
    PointLights,
    DirectionalLights,
    Materials,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    # SoftPhongShader,
    HardPhongShader,
    # TexturesUV,
    # BlendParams,
    # SoftSilhouetteShader,
    # materials
)
import torch
from torch.utils.data import Dataset, DataLoader
import os
import cv2
import numpy as np
import warnings

warnings.filterwarnings("ignore")


class MyDataset(Dataset):
    def __init__(self, mesh, data_dir, img_size, device=''):
        # 初始化数据目录、文件列表、图像尺寸和设备
        self.data_dir = data_dir
        self.files = []

        # 加载数据目录下的所有文件名
        files = os.listdir(data_dir)
        for file in files:
            self.files.append(file)
        print(f"加载 {len(self.files)} 个文件")  # 打印文件数量

        self.img_size = img_size
        self.device = device
        self.mesh = mesh

        # 设置渲染参数
        raster_settings = RasterizationSettings(
            image_size=self.img_size,
            blur_radius=0.0,
            faces_per_pixel=1,
            max_faces_per_bin=250000  # 最大每bin的面数
        )

        # 初始化光源
        lights = PointLights(device=self.device, location=[[100.0, 85, 100.0]])

        # 初始化相机
        self.cameras = ''

        # 初始化渲染器
        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=self.cameras,
                raster_settings=raster_settings
            ),
            shader=HardPhongShader(
                device=self.device,
                cameras=self.cameras,
                lights=lights
            )
        )

    def set_cameras(self, cameras):
        self.cameras = cameras  #

    def set_mesh(self, mesh):
        self.mesh = mesh

    def __getitem__(self, index):
        """
        根据提供的索引值返回数据集中的特定样本

        参数:
            index (int): 索引值，用于标识数据集中的特定文件

        返回值:
            tuple: 包含索引、文件名、对抗样本图像、预测图像和掩码的元组
        """
        # 根据索引值和数据集目录获取文件名
        file = os.path.join(self.data_dir, self.files[index])
        # 加载.npz 文件中的数据
        data = np.load(file)
        # 获取 'img' 键对应的值作为图像数据
        img = data['img']
        # 获取 'veh_trans' 键对应的值作为车辆转换信息
        veh_trans = data['veh_trans']
        # 获取 'cam_trans' 键对应的值作为相机转换信息
        cam_trans = data['cam_trans']
        # 获取文件名（不包括扩展名）
        # file_name = file.split('/')[-1].split('.npz')[0]

        # 对相机转换信息进行缩放
        scale = 150
        for i in range(0, 3):
            cam_trans[0][i] = cam_trans[0][i] * scale

        # 获取摄像机的参数（眼睛位置、摄像机方向、摄像机向上方向）
        eye, camera_direction, camera_up = get_params(cam_trans, veh_trans)

        # 计算视角变换（眼睛位置、向上方向、焦点）
        # R为旋转矩阵，T 为平移矩阵
        R, T = look_at_view_transform(eye=(tuple(eye),), up=(tuple(camera_up),), at=((0, 0, 0),))
        # 反转矩阵的第一个维度元素的符号
        R[:, :, 0] = R[:, :, 0] * -1
        # 反转矩阵的第二维度元素的符号
        R[:, 0, :] = R[:, 0, :] * -1
        # 交换矩阵的第二和第三条维度
        tmp = R[:, 1, :].clone()
        R[:, 1, :] = R[:, 2, :].clone()
        R[:, 2, :] = tmp

        # 创建一个具有给定参数的透视摄像机对象
        train_cameras = FoVPerspectiveCameras(device=self.device, R=R, T=T, znear=1.0, zfar=300.0, fov=45.0)

        # 计算摄像机的观察方向
        direction = list(1 * np.array(
            torch.bmm(R, torch.from_numpy(np.array(camera_direction)).unsqueeze(0).unsqueeze(2).float()).squeeze()))
        # 设置渲染器的灯光为定向灯光，方向由摄像机方向决定
        self.renderer.shader.lights = DirectionalLights(device=self.device, direction=[direction])

        # 创建一个具有指定设备的材质对象，用于渲染
        materials = Materials(
            device=self.device,
            specular_color=[[1.0, 1.0, 1.0]],
            shininess=500.0
        )

        # 设置渲染器的摄像机为训练摄像机
        self.renderer.rasterizer.cameras = train_cameras
        # 设置渲染器的着色器的摄像机为训练摄像机
        self.renderer.shader.cameras = train_cameras

        # 使用渲染器渲染模型，并获取渲染后的图像
        images = self.renderer(self.mesh, materials=materials)
        # 从渲染后的图像中提取颜色信息作为预测图像
        imgs_pred = images[:, ..., :3]

        # 交换图像的颜色通道顺序，从 RGB 变为 BGR，这是因为 OpenCV 默认的颜色通道顺序是 BGR
        img = img[:, :, ::-1]
        # 使用 OpenCV 的 resize 函数调整图像大小
        img_cv = cv2.resize(img, (self.img_size, self.img_size))
        # 调整图像维度顺序，从 HWC 变为 CHW，这是 PyTorch 模型期望的输入格式
        img = np.transpose(img_cv, (2, 0, 1))
        # 调整图像大小，使其具有四个维度，以适配 PyTorch 模型的输入形状
        img = np.resize(img, (1, img.shape[0], img.shape[1], img.shape[2]))
        # 将 numpy 数组转换为 PyTorch 张量
        img = torch.from_numpy(img).float().to(self.device)
        # 归一化图像张量，将像素值从 0-255 映射到 0.0-1.0
        img /= 255.0

        # 背景形状（即原始图像的形状）
        bg_shape = img.shape
        # 汽车尺寸（即渲染图像的尺寸）
        car_size = self.renderer.rasterizer.raster_settings.image_size
        # 计算宽和高的填充量，用于在原始图像中为渲染图像创建一个中心位置
        dH = bg_shape[2] - car_size
        dW = bg_shape[3] - car_size
        location = (
            dW // 2,
            dW - (dW // 2),
            dH // 2,
            dH - (dH // 2)
        )
        # 创建一个掩码，用于标记渲染图像中的汽车区域，值为 1 的像素对应于背景
        contour = torch.where((imgs_pred == 1), torch.zeros(1).to(self.device), torch.ones(1).to(self.device))
        # 创建一个新的掩码，其形状与原始图像相同，用于记录填充后的汽车区域
        new_contour = torch.zeros(img.permute(0, 2, 3, 1).shape, device=self.device)
        # 对掩码进行填充操作，使其尺寸与图像张量相同
        new_contour[:, :, :, 0] = F.pad(contour[:, :, :, 0], location, "constant", value=0)
        new_contour[:, :, :, 1] = F.pad(contour[:, :, :, 1], location, "constant", value=0)
        new_contour[:, :, :, 2] = F.pad(contour[:, :, :, 2], location, "constant", value=0)

        # 创建一个新的张量，其形状与原始图像相同，用于记录填充后的渲染图像
        new_car = torch.zeros(img.permute(0, 2, 3, 1).shape, device=self.device)
        # 对渲染图像进行填充操作，使其尺寸与图像张量相同
        new_car[:, :, :, 0] = F.pad(imgs_pred[:, :, :, 0], location, "constant", value=0)
        new_car[:, :, :, 1] = F.pad(imgs_pred[:, :, :, 1], location, "constant", value=0)
        new_car[:, :, :, 2] = F.pad(imgs_pred[:, :, :, 2], location, "constant", value=0)

        # 创建对抗样本图像，将掩码为 0 的区域填充为原始图像，掩码为 1 的区域填充为渲染图像
        total_img = torch.where((new_contour == 0.), img.permute(0, 2, 3, 1), new_car)

        # 返回索引、文件名、对抗样本图像、预测图像和掩码
        return index, file, total_img.squeeze(0), imgs_pred.squeeze(0), new_contour.squeeze(0)

    def __len__(self):
        return len(self.files)


# def initialize_patch(mesh, device, texture_atlas_size):
#     # 打印 'Initializing patch...' 字符串，表示正在初始化面片
#     print('Initializing patch...')
#
#     # 初始化一个空列表 sampled_planes，用于存储采样的平面
#     sampled_planes = list()
#
#     # 打开名为 'top_faces_QZH.txt' 的文本文件，读取其中的每一行
#     with open(r'top_faces.txt', 'r') as f:
#
#         # 读取文件中的每一行，将其作为 face_id
#         face_ids = f.readlines()
#
#         # 遍历 face_ids 列表中的每个 face_id
#         for face_id in face_ids:
#
#             # 如果 face_id 不是空行
#             if face_id != '\n':
#                 # 将 face_id 转换为整数，并添加到 sampled_planes 列表中
#                 sampled_planes.append(int(face_id))
#
#     # 使用 sampled_planes 列表创建一个 PyTorch 张量 idx，数据类型为 long，并移动到指定设备上
#     idx = torch.Tensor(sampled_planes).long().to(device)
#
#     # 创建一个形状为 (len(sampled_planes), texture_atlas_size, texture_atlas_size, 3) 的随机张量 patch
#     # 其中 len(sampled_planes) 表示采样平面的数量，texture_atlas_size 表示纹理图谱的大小，3 表示颜色通道
#     # 将 patch 张量移动到指定设备上，并设置 requires_grad=True，表示这个张量需要计算梯度，以便于后续的优化
#     patch = torch.rand(len(sampled_planes), texture_atlas_size, texture_atlas_size, 3, device=device,
#                        requires_grad=True)
#
#     # 返回初始化后的面片 patch 和索引 idx
#     return patch, idx
