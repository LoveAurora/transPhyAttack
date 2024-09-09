from math import pi
import torch
from torch.utils.data import Dataset, DataLoader
import os
import sys
import cv2
import numpy as np
import warnings

warnings.filterwarnings("ignore")
from torchvision.utils import save_image
from PIL import Image
from MeshDataset import MeshDataset
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights,
    DirectionalLights,
    Materials,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    HardPhongShader,
)
import torch.nn.functional as F

from utils import get_params

# 设置环境变量
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# 确认选择的设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# 渲染3D网格模型到纹理，之后将渲染的纹理贴到车上，并将结果与背景图像融合。

class MyDataset(Dataset):
    def __init__(self, mesh, data_dir, img_size, device=''):
        # 设置数据目录
        self.data_dir = data_dir
        # 初始化文件列表
        self.files = []
        # 获取数据目录下的所有文件
        files = os.listdir(data_dir)
        for file in files:
            # 将文件添加到文件列表中
            self.files.append(file)
        # 打印文件总数，用于检查
        print(len(self.files))
        # 设置图像大小
        self.img_size = img_size
        # 设置设备（CPU或GPU）
        self.device = device
        # 设置网格模型
        self.mesh = mesh
        # 设置光栅化设置
        raster_settings = RasterizationSettings(
            image_size=self.img_size,  # 图像大小
            blur_radius=0.0,  # 模糊半径
            faces_per_pixel=1,  # 每像素的面数
            # bin_size=0#  # 注释掉的参数，可能用于优化
            max_faces_per_bin=250000  # 每个bin的最大面数
        )

        # 设置灯光
        lights = PointLights(device=self.device, location=[[100.0, 85, 100.0]])
        # 初始化相机
        self.cameras = ''
        # 初始化渲染器
        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=self.cameras,  # 相机
                raster_settings=raster_settings  # 光栅化设置
            ),
            shader=HardPhongShader(  # 使用硬Phong着色器
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
        # 根据索引构建数据文件的完整路径
        file = os.path.join(self.data_dir, self.files[index])
        # 从文件中加载数据
        data = np.load(file)
        # 从加载的数据中提取图像、车辆变换和相机变换
        img = data['img']
        veh_trans = data['veh_trans']
        cam_trans = data['cam_trans']
        # 提取不包含目录和文件扩展名的文件名
        file_name = file.split('/')[-1].split('.npz')[0]
        # print(file_name)
        # 定义相机变换的缩放因子
        scale = 215
        # 缩放相机变换
        for i in range(0, 3):
            cam_trans[0][i] = cam_trans[0][i] * scale

        # 根据车辆和相机变换计算相机参数
        eye, camera_direction, camera_up = get_params(cam_trans, veh_trans)

        # 为相机生成旋转和平移矩阵
        R, T = look_at_view_transform(eye=(tuple(eye),), up=(tuple(camera_up),), at=((0, 0, 0),))
        # 调整旋转矩阵以匹配预期的坐标系统
        R[:, :, 0] = R[:, :, 0] * -1
        R[:, 0, :] = R[:, 0, :] * -1
        tmp = R[:, 1, :].clone()
        R[:, 1, :] = R[:, 2, :].clone()
        R[:, 2, :] = tmp

        # 使用计算出的旋转和平移矩阵初始化相机
        train_cameras = FoVPerspectiveCameras(device=self.device, R=R, T=T, znear=1.0, zfar=300.0, fov=45.0)
        # 根据相机方向计算光线方向
        direction = list(1 * np.array(
            torch.bmm(R, torch.from_numpy(np.array(camera_direction)).unsqueeze(0).unsqueeze(2).float()).squeeze()))
        # 在着色器中设置光线方向
        self.renderer.shader.lights = DirectionalLights(device=self.device, direction=[direction])
        # 定义材质属性
        materials = Materials(
            device=self.device,
            ambient_color=[[1.0, 1.0, 1.0]],  # 仅设置 RGB
            diffuse_color=[[1.0, 1.0, 1.0]],  # 仅设置 RGB

            specular_color=[[1.0, 1.0, 1.0]],  # 仅设置 RGB
            shininess=500.0
        )
        # 修改渲染器的 blend_params
        # blend_params = BlendParams(sigma=1e-4, gamma=1e-4, background_color=(0.0, 0.0, 0.0, 0.0))
        # self.renderer.shader.blend_params = blend_params
        # 在光栅化器和着色器中设置相机
        self.renderer.rasterizer.cameras = train_cameras
        self.renderer.shader.cameras = train_cameras
        # 使用当前的网格和材质设置渲染图像
        images = self.renderer(self.mesh, materials=materials)
        imgs_pred = images[:, ..., :3]

        #
        img_save_dir = 'RenderedTextures/'
        os.makedirs(img_save_dir, exist_ok=True)
        rendered_img_path = os.path.join(img_save_dir, file_name + '_rendered.png')

        imgs_pred_np = imgs_pred.squeeze(0).cpu().detach().numpy()
        if imgs_pred_np.dtype != np.uint8:
            imgs_pred_np = (imgs_pred_np * 255).astype(np.uint8)
        if imgs_pred_np.shape[0] == 3:
            imgs_pred_np = np.transpose(imgs_pred_np, (1, 2, 0))

        # Image.fromarray(imgs_pred_np).save(rendered_img_path)



        # 将图像从BGR格式转换为RGB格式
        img = img[:, :, ::-1]
        # 将图像调整到指定大小
        img_cv = cv2.resize(img, (self.img_size, self.img_size))
        # 重新排列图像维度并调整大小以进行批处理
        img = np.transpose(img_cv, (2, 0, 1))
        img = np.resize(img, (1, img.shape[0], img.shape[1], img.shape[2]))
        # 将图像转换为PyTorch张量并移动到指定设备
        img = torch.from_numpy(img).cuda(device=0).float()
        # 将图像值归一化到范围[0, 1]
        img /= 255.0

        # 计算背景和汽车图像之间的高度和宽度差
        bg_shape = img.shape
        car_size = self.renderer.rasterizer.raster_settings.image_size
        dH = bg_shape[2] - car_size
        dW = bg_shape[3] - car_size
        # 定义填充位置
        location = (
            dW // 2,
            dW - (dW // 2),
            dH // 2,
            dH - (dH // 2)
        )
        # 生成渲染汽车的轮廓
        contour = torch.where((imgs_pred == 1), torch.zeros(1).to(self.device), torch.ones(1).to(self.device))
        # 使用填充初始化新的轮廓张量
        new_contour = torch.zeros(img.permute(0, 2, 3, 1).shape, device=self.device)
        new_contour[:, :, :, 0] = F.pad(contour[:, :, :, 0], location, "constant", value=0)
        new_contour[:, :, :, 1] = F.pad(contour[:, :, :, 1], location, "constant", value=0)
        new_contour[:, :, :, 2] = F.pad(contour[:, :, :, 2], location, "constant", value=0)

        # 使用填充初始化新的汽车图像张量
        new_car = torch.zeros(img.permute(0, 2, 3, 1).shape, device=self.device)
        new_car[:, :, :, 0] = F.pad(imgs_pred[:, :, :, 0], location, "constant", value=0)
        new_car[:, :, :, 1] = F.pad(imgs_pred[:, :, :, 1], location, "constant", value=0)
        new_car[:, :, :, 2] = F.pad(imgs_pred[:, :, :, 2], location, "constant", value=0)

        # 根据轮廓将新的汽车图像与原始图像结合
        total_img = torch.where((new_contour == 0.), img.permute(0, 2, 3, 1), new_car)

        img_save_dir = 'EvaluationImg/'
        os.makedirs(img_save_dir, exist_ok=True)

        # 确保 imgs_pred 是 uint8 格式，并且形状正确
        imgs_pred_np = imgs_pred.squeeze(0).cpu().detach().numpy()
        if imgs_pred_np.dtype != np.uint8:
            imgs_pred_np = (imgs_pred_np * 255).astype(np.uint8)
        if imgs_pred_np.shape[0] == 3:
            imgs_pred_np = np.transpose(imgs_pred_np, (1, 2, 0))

        rendered_img_path = os.path.join(img_save_dir, file_name + '_rendered.jpg')
        # Image.fromarray(imgs_pred_np).save(rendered_img_path)

        save_image(total_img[0, :, :, :].unsqueeze(0).permute(0, 3, 1, 2).cpu().detach(),
                   img_save_dir + file_name + '.jpg')

        # 打印张量形状
        print(
            f'index: {index}, total_img shape: {total_img.shape}, imgs_pred shape: {imgs_pred.shape}, new_contour shape: {new_contour.shape}')

        return index, file, total_img.squeeze(0), imgs_pred.squeeze(0), new_contour.squeeze(0)

    def __len__(self):
        return len(self.files)


if __name__ == '__main__':
    import tqdm
    import argparse

    # 创建命令行参数解析器
    parser = argparse.ArgumentParser()
    # 检查是否有可用的CUDA设备，如果有，使用第一个CUDA设备
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    # 添加命令行参数
    parser.add_argument('--mesh_dir', type=str, default=r"3d_model")  # 网格模型目录
    parser.add_argument('--patch_dir', type=str, default='')  # 补丁目录
    parser.add_argument('--img_size', type=int, default=608)  # 图像大小
    parser.add_argument('--test_dir', type=str, default=r'./dataset/testset/')  # 测试集目录
    parser.add_argument('--texture_atlas_size', type=int, default=1)  # 纹理图集大小
    config = parser.parse_args()

    texture_atlas_size = config.texture_atlas_size
    # 加载网格数据集
    mesh_dataset = MeshDataset(config.mesh_dir, device, texture_atlas_size=texture_atlas_size, max_num=1)
    for mesh in mesh_dataset:
        # 加载补丁和索引
        patch = torch.load(config.patch_dir + 'patch_save.pt').to(device)
        idx = torch.load(config.patch_dir + 'idx_save.pt').to(device)

        # 获取纹理图像
        texture_image = mesh.textures.atlas_padded()

        # 限制补丁值的范围
        clamped_patch = patch.clone().clamp(min=1e-6, max=0.99999)
        # 更新纹理图集
        mesh.textures._atlas_padded[:, idx, :, :, :] = clamped_patch
        mesh.textures.atlas = mesh.textures._atlas_padded
        mesh.textures._atlas_list = None
        # 创建数据集和数据加载器
        dataset = MyDataset(mesh, config.test_dir, int(config.img_size), device=device)
        loader = DataLoader(
            dataset=dataset,
            batch_size=1,
            shuffle=False,
        )
        tqdm_loader = tqdm.tqdm(loader)
        img_save_dir = 'SavedTextures/'
        os.makedirs(img_save_dir, exist_ok=True)
        for i, (index, file_name, total_img, texture_img, contour) in enumerate(tqdm_loader):
            # 定义保存纹理图像和轮廓图像的路径
            file_name_with_ext = file_name[0]
            file_name_new = os.path.basename(file_name_with_ext).split('.npz')[0]
            texture_img_path = os.path.join(img_save_dir, f"{file_name_new}.jpg")
            contour_img_path = os.path.join(img_save_dir, f"{file_name_new}_contour.jpg")

            print(f"Texture image shape before processing: {texture_img.shape}")

            # 调整纹理图像和轮廓图像的维度
            if texture_img.dim() == 4:
                texture_img = texture_img.permute(0, 3, 1, 2)  # Change to (N, C, H, W)
            elif texture_img.dim() == 3:
                texture_img = texture_img.permute(2, 0, 1).unsqueeze(0)  # Change to (N, C, H, W)
            else:
                raise ValueError("texture_img has incorrect number of dimensions")

            if contour.dim() == 4:
                contour = contour.permute(0, 3, 1, 2)  # Change to (N, C, H, W)
            elif contour.dim() == 3:
                contour = contour.permute(2, 0, 1).unsqueeze(0)  # Change to (N, C, H, W)
            else:
                raise ValueError("texture_img has incorrect number of dimensions")

            print(f"Texture image shape after processing: {texture_img.shape}")
            # 如果在GPU上，将张量移动到CPU
            texture_img = texture_img.cpu()
            contour = contour.cpu()
            # 确保张量值在[0, 1]范围内
            texture_img = texture_img.clamp(0, 1)
            contour = contour.clamp(0, 1)
            # 保存图像
            # save_image(contour.squeeze(0), contour_img_path)
            # save_image(texture_img.squeeze(0), texture_img_path)
