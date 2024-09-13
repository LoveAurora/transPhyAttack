import os
import torch
from torch.utils.data import DataLoader
from MeshDataset import MeshDataset
from loss import TotalVariation_3d, MaxProbExtractor, NPSCalculator
import time
from tqdm import tqdm
from YoloV3.yolov3_CAM import YOLO
from data_loader_nr import MyDataset


class Patch:
    def __init__(self, config, device):
        # 初始化配置和设备
        self.config = config
        self.device = device

        # 初始化网格数据集
        self.mesh_dataset = MeshDataset(
            config.mesh_dir,
            device,
            texture_atlas_size=config.texture_atlas_size,
            max_num=config.num_meshes
        )

        # 初始化YOLO模型
        self.dnet = YOLO(config)

        # 初始化概率提取器
        self.prob_extractor = MaxProbExtractor(cls_id=0, num_cls=1, config=self.config).cuda()

        # 初始化对抗补丁
        self.patch = None
        self.idx = None

        # 如果指定了补丁目录，则加载补丁和索引
        if self.config.patch_dir is not None:
            self.patch = torch.load(self.config.patch_dir + 'patch_save.pt').to(self.device)
            self.patch.requires_grad = True
            self.idx = torch.load(self.config.patch_dir + 'idx_save.pt').to(self.device)

        # 如果补丁或索引为空，则初始化新的补丁
        if self.patch is None or self.idx is None:
            self.initialize_patch(device=self.device, texture_atlas_size=config.texture_atlas_size)

        # 初始化NPS计算器
        self.nps_calculator = NPSCalculator(self.config.printfile, self.patch.shape).cuda()

        # 设置对比度和亮度调整参数
        self.min_contrast = 0.9
        self.max_contrast = 1.1
        self.min_brightness = -0.1
        self.max_brightness = 0.1
        self.noise_factor = 0.10

    def attack(self):
        # 获取初始网格模型
        mesh = self.mesh_dataset.meshes[0]
        # 初始化三维总变差损失函数并移动到指定设备
        total_variation = TotalVariation_3d(mesh, self.idx).to(self.device)
        # 定义优化器
        optimizer = torch.optim.SGD([self.patch], lr=1e-2, momentum=0.9)
        # 设置训练轮数
        n_epochs = self.config.epochs
        # 开始训练循环
        for epoch in range(n_epochs):
            # 遍历数据集中的每个网格模型
            for mesh in self.mesh_dataset:
                # 复制并限制补丁的值范围
                clamped_patch = self.patch.clone().clamp(min=1e-6, max=0.99999)
                # 更新网格纹理
                mesh.textures._atlas_padded[:, self.idx, :, :, :] = clamped_patch
                mesh.textures.atlas = mesh.textures._atlas_padded
                mesh.textures._atlas_list = None
                # 创建自定义数据集实例
                dataset = MyDataset(mesh, self.config.train_dir, self.config.img_size, device=self.device)
                # 初始化数据加载器
                loader = DataLoader(
                    dataset=dataset,
                    batch_size=self.config.batch_size,
                    shuffle=self.config.shuffle,
                    drop_last=self.config.drop_last,
                )
                # 使用tqdm显示进度条
                tqdm_loader = tqdm(loader)
                # 遍历数据加载器中的每个批次
                for i, (index, file_name_batch_size, total_img, texture_img, contour) in enumerate(tqdm_loader):

                    # 清零梯度
                    optimizer.zero_grad()
                    # 调整图像维度顺序
                    total_img = total_img.permute(0, 3, 1, 2)  # [N H W C]->[N C H W]
                    # 获取网络输出
                    output = self.dnet.get_output(total_img)

                    # -----------------------------averaged multi-scale attention map---------------------#
                    # 注册hook以获取多尺度注意力图
                    inputs = {"image": total_img}
                    self.dnet.multi_attention.register_hook()
                    attention_list, _, _ = self.dnet.multi_attention(inputs, retain_graph=True)
                    self.dnet.multi_attention.remove_handlers()
                    # ------------------------------------------------------------------------------------#
                    # 初始化前景和背景注意力列表
                    heatmap_constrain_list = list()  # 前景注意力
                    heatmap_background_list = list()  # 背景注意力
                    heatmap_constrain = None
                    heatmap_background = None
                    # 计算每个尺度的注意力图
                    for j in range(len(attention_list)):
                        heatmap_constrain_temp = attention_list[j] * contour.permute(0, 3, 1, 2)
                        heatmap_constrain_list.append(heatmap_constrain_temp)

                        heatmap_background_temp = attention_list[j] - heatmap_constrain_temp
                        heatmap_background_list.append(heatmap_background_temp)
                        # 累加注意力图
                        if j == 0:
                            heatmap_constrain = heatmap_constrain_temp
                            heatmap_background = heatmap_background_temp
                        else:
                            heatmap_constrain = heatmap_constrain + heatmap_constrain_temp  # accumulate foreground attention
                            heatmap_background = heatmap_background + heatmap_background_temp  # accumulate background attention
                    # 计算全局平均注意力值
                    heat_average = (torch.sum(heatmap_constrain, dim=[1, 2, 3]) /
                                    torch.sum(heatmap_constrain != 0, dim=[1, 2, 3]))
                    # 计算注意力损失
                    heat_average_bg = (torch.sum(heatmap_background, dim=[1, 2, 3]) /
                                       torch.sum(heatmap_constrain == 0, dim=[1, 2, 3]))
                    # global average value of background attention

                    heat_loss = torch.mean(heat_average) * 5 - torch.mean(heat_average_bg)  # attention loss
                    tv_loss = total_variation(self.patch) * 2.5
                    nps = self.nps_calculator(self.patch)
                    # 计算总损失
                    loss = heat_loss * 1 + tv_loss * 1 + nps * 1
                    loss.backward(retain_graph=True)
                    optimizer.step()
                    # 日志记录
                    log_dir = ''
                    with open(os.path.join(log_dir, 'loss.txt'), 'a') as f:
                        tqdm_loader.set_description('Epoch %d/%d ,Loss %.3f,heat_loss_new %.3f,tv_loss %.3f,nps %.3f ' % \
                                                    (epoch, n_epochs, loss.data.cpu().numpy(),
                                                     heat_loss.data.cpu().numpy(), tv_loss.data.cpu().numpy(),
                                                     nps.data.cpu().numpy()))

                        if i == 0:
                            f.write(time.strftime("%Y%m%d-%H%M%S") + '\n')
                        f.write('Epoch %d/%d ,Loss %.3f,heat_loss_new %.3f,tv_loss %.3f,nps %.3f \n' % \
                                (epoch, n_epochs, loss.data.cpu().numpy(), heat_loss.data.cpu().numpy(),
                                 tv_loss.data.cpu().numpy(), nps.data.cpu().numpy()))

                    # 随机增强补丁
                    contrast = torch.FloatTensor(1).uniform_(self.min_contrast, self.max_contrast).to(self.device)
                    brightness = torch.FloatTensor(1).uniform_(self.min_brightness, self.max_brightness).to(self.device)
                    noise = torch.FloatTensor(self.patch.shape).uniform_(-1, 1) * self.noise_factor
                    noise = noise.to(self.device)
                    augmented_patch = (self.patch * contrast) + brightness + noise

                    # 限制增强后的补丁值范围
                    clamped_patch = augmented_patch.clone().clamp(min=1e-6, max=0.99999)
                    mesh.textures._atlas_padded[:, self.idx, :, :, :] = clamped_patch
                    mesh.textures.atlas = mesh.textures._atlas_padded
                    mesh.textures._atlas_list = None

                    dataset.set_mesh(mesh)

                    del output, total_img, texture_img, contour, nps, loss, tv_loss, attention_list
                    torch.cuda.empty_cache()

            patch_save = self.patch.cpu().detach().clone()
            idx_save = self.idx.cpu().detach().clone()
            torch.save(patch_save, 'patch_save_bwm.pt')
            torch.save(idx_save, 'idx_save_bwm.pt')

    def initialize_patch(self, device, texture_atlas_size):
        # 打印初始化补丁的消息
        print('Initializing patch...')

        # 创建一个空列表来存储采样的平面
        sampled_planes = list()

        # 打开文件'top_faces.txt'并读取所有行
        with open(r'top_faces.txt', 'r') as f:
            face_ids = f.readlines()

            # 遍历每一行，如果行不为空，则将其转换为整数并添加到采样平面列表中
            for face_id in face_ids:
                if face_id != '\n':
                    sampled_planes.append(int(face_id))

        # 将采样平面列表转换为张量并移动到指定设备
        idx = torch.Tensor(sampled_planes).long().to(device)

        # 创建一个随机初始化的补丁张量，形状为(采样平面数量, 纹理图集大小, 纹理图集大小, 3)，并设置requires_grad为True
        patch = torch.rand(len(sampled_planes), texture_atlas_size, texture_atlas_size, 3, device=(device),
                           requires_grad=True)

        # 初始化self.idx
        self.idx = idx

        # 初始化self.patch
        self.patch = patch


def main():
    import argparse
    import sys
    parser = argparse.ArgumentParser()
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    parser.add_argument('--mesh_dir', type=str, default=r"3d_model")
    parser.add_argument('--patch_dir', type=str, default=None,
                        help='patch_dir is None normally, but it should be a certain path when resuming texture optimization from the last epoch')
    # parser.add_argument('--patch_dir', type=str, default='',help='patch_dir is None normally, but it should be a  certain path when resuming texture optimization from the last epoch')
    # parser.add_argument('--idx', type=str, default='')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--img_size', type=int, default=608)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--shuffle', type=bool, default=False, help='whether shuffle the data when training')
    parser.add_argument('--drop_last', type=bool, default=False)
    parser.add_argument('--num_meshes', type=int, default=1)
    parser.add_argument('--texture_atlas_size', type=int, default=1)
    parser.add_argument('--detector', type=str, default='yolov3')
    parser.add_argument('--conf_thres', type=int, default=0.25, help='conf_thres of yolov3')
    parser.add_argument('--iou_thres', type=int, default=0.5, help='iou_thres of yolov3')
    parser.add_argument('--printfile', type=str, default=r'non_printability/30values.txt')
    parser.add_argument('--train_dir', type=str, default=r'dataset/trainset')  # 训练目录
    parser.add_argument('--weightfile', type=str,
                        default=r"YoloV3/checkpoint/Epoch300-Total_Loss1.9337-Val_Loss1.9373.pth")  # 权重文件路径
    config = parser.parse_args()
    trainer = Patch(config, device)

    if config.detector == 'yolov3':
        trainer.attack()


if __name__ == '__main__':
    main()
