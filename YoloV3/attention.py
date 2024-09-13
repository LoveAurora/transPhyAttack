import cv2
from YoloV3.misc_functions import *
import torchvision.transforms as transforms

import warnings
from torchvision.transforms.functional import InterpolationMode
from YoloV3.myRandomAffine import myRandomAffine


class Attention(object):

    def __init__(self, net, ori_shape, final_shape, yolo_decodes, num_classes, conf_thres, nms_thres):
        """
        初始化YOLO检测器。

        参数:
            net (torch.nn.Module): YOLO网络模型。
            ori_shape (tuple): 原始输入图像的形状。
            final_shape (tuple): 最终处理后的图像形状。
            yolo_decodes (list): YOLO解码器列表。
            num_classes (int): 类别数量。
            conf_thres (float): 置信度阈值。
            nms_thres (float): 非极大值抑制阈值。
        """
        self.net = net  # 保存YOLO网络模型
        self.ori_shape = ori_shape  # 保存原始输入图像的形状
        self.final_shape = final_shape  # 保存最终处理后的图像形状
        self.feature = list()  # 用于存储特征信息的列表
        self.gradient = list()  # 用于存储梯度信息的列表
        self.net.eval()  # 将网络设置为评估模式
        self.yolo_decodes = yolo_decodes  # 保存YOLO解码器列表
        self.num_classes = num_classes  # 保存类别数量
        self.conf_thres = conf_thres  # 保存置信度阈值
        self.nms_thres = nms_thres  # 保存非极大值抑制阈值
        self.handlers = []  # 用于存储回调处理器的列表

    def _get_features_hook(self, module, input, output):
        """
        用于捕获并存储中间层特征的钩子函数。

        参数:
            module (torch.nn.Module): 当前模块。
            input (tuple): 输入数据。
            output (torch.Tensor): 输出数据。

        说明:
            - 将输出数据追加到 `self.feature` 列表中。
            - 打印输出数据的形状（可选）。
        """
        self.feature.append(output)  # 将输出数据追加到特征列表中
        # print("feature shape:{}".format(output.size()))  # 打印输出数据的形状（可选）

    def _get_grads_hook(self, module, input_grad, output_grad):
        """
        用于捕获并存储中间层梯度的钩子函数。

        参数:
            module (torch.nn.Module): 当前模块。
            input_grad (tuple): 输入梯度。
            output_grad (tuple): 输出梯度。

        说明:
            - 将输出梯度的第一个元素追加到 `self.gradient` 列表中。
            - 打印输出梯度的形状（可选）。
        """
        self.gradient.append(output_grad[0])  # 将输出梯度的第一个元素追加到梯度列表中
        # print('output_grad[0].shape=', output_grad[0].shape)  # 打印输出梯度的形状（可选）

    def register_hook(self):
        """
        注册前向和后向钩子函数，用于捕获特征和梯度。

        说明:
            - 清空 `self.feature`, `self.gradient` 和 `self.handlers`。
            - 遍历网络中的特定模块，并注册前向和后向钩子函数。
        """
        self.feature = list()  # 清空特征列表
        self.gradient = list()  # 清空梯度列表
        self.handlers = []  # 清空回调处理器列表

        # 注册 layer5.residual_1.conv2 的前向和后向钩子
        for name, module in self.net.module.backbone.layer5.residual_1._modules.items():
            if module == self.net.module.backbone.layer5.residual_1.conv2:
                self.handlers.append(module.register_forward_hook(self._get_features_hook))
                self.handlers.append(module.register_backward_hook(self._get_grads_hook))

        # 注册 layer4.residual_5.conv2 的前向和后向钩子
        for name, module in self.net.module.backbone.layer4.residual_5._modules.items():
            if module == self.net.module.backbone.layer4.residual_5.conv2:
                self.handlers.append(module.register_forward_hook(self._get_features_hook))
                self.handlers.append(module.register_backward_hook(self._get_grads_hook))

        # 注册 layer3.residual_2.conv2 的前向和后向钩子
        for name, module in self.net.module.backbone.layer3.residual_2._modules.items():
            if module == self.net.module.backbone.layer3.residual_2.conv2:
                self.handlers.append(module.register_forward_hook(self._get_features_hook))
                self.handlers.append(module.register_backward_hook(self._get_grads_hook))

    def remove_handlers(self):
        for handle in self.handlers:
            handle.remove()
            torch.cuda.empty_cache()

    def __call__(self, inputs, index=0, retain_graph=True):

        img_origin = inputs['image']
        # 创建一个随机水平翻转对象Hflip
        Hflip = transforms.RandomHorizontalFlip(p=1)
        # randomAffine  创建随机仿射变换对象，返回仿射变换后的图像，以及平移量和缩放系数。
        # img_randomAffine  为经过randomAffine变换后的图像。
        # inverse_randomAffine  创建一个逆向的随机仿射变换对象，用于撤销randomAffine变换。
        # translations 存储 randomAffine 变换后的图像相对于原始图像的平移量。
        # scale_factor 存储 randomAffine_scale 变换后的图像相对于randomAffine 变换后的图像的缩放因子。
        # img_randomAffineScale  为经过randomAffine_scale变换后的图像。
        # scale_randomAffine  创建随机缩放变换对象，返回缩放变换后的图像，以及平移量和缩放系数。
        # inverse_randomScale  创建一个逆向的随机缩放变换对象，用于撤销randomAffine_scale变换。

        '''第一个复合变换：平移->缩放'''
        # 平移
        randomAffine1 = myRandomAffine(degrees=(0, 0), translate=(0.1, 0.1))
        img_randomAffine1, translations1, _ = randomAffine1(img_origin.clone())
        # 逆平移
        inverse_randomAffine1 = myRandomAffine(degrees=(0, 0), inverse_translate=(-translations1[0], -translations1[1]))

        # 缩放
        scale_randomAffine1 = myRandomAffine(degrees=(0, 0), scale=(0.8, 1.2), interpolation=InterpolationMode.BILINEAR)
        img_randomAffineScale1, _, scale_factor1 = scale_randomAffine1(img_randomAffine1)
        # 逆缩放
        inverse_randomScale1 = myRandomAffine(degrees=(0, 0), scale=(1 / scale_factor1, 1 / scale_factor1),
                                            interpolation=InterpolationMode.BILINEAR)

        '''第二个：水平翻转->平移->缩放'''
        # 水平翻转
        img_flip2 = Hflip(img_origin.clone())
        # 平移
        randomAffine2 = myRandomAffine(degrees=(0, 0), translate=(0.1, 0.1))
        img_randomAffine2, translations2, _ = randomAffine2(img_flip2)
        # 逆平移
        inverse_randomAffine2 = myRandomAffine(degrees=(0, 0), inverse_translate=(-translations2[0], -translations2[1]))

        # 缩放
        scale_randomAffine2 = myRandomAffine(degrees=(0, 0), scale=(0.8, 1.2), interpolation=InterpolationMode.BILINEAR)
        img_randomAffineScale2, _, scale_factor2 = scale_randomAffine2(img_randomAffine2)
        # 逆缩放
        inverse_randomScale2 = myRandomAffine(degrees=(0, 0), scale=(1 / scale_factor2, 1 / scale_factor2),
                                            interpolation=InterpolationMode.BILINEAR)

        '''第三个：平移->缩放'''
        # 平移
        randomAffine3 = myRandomAffine(degrees=(0, 0), translate=(0.1, 0.1))
        img_randomAffine3, translations3, _ = randomAffine3(img_origin.clone())
        # 逆平移
        inverse_randomAffine3 = myRandomAffine(degrees=(0, 0), inverse_translate=(-translations3[0], -translations3[1]))

        # 缩放
        scale_randomAffine3 = myRandomAffine(degrees=(0, 0), scale=(0.8, 1.2), interpolation=InterpolationMode.BILINEAR)
        img_randomAffineScale3, _, scale_factor3 = scale_randomAffine3(img_randomAffine3)
        # 逆缩放
        inverse_randomScale3 = myRandomAffine(degrees=(0, 0), scale=(1 / scale_factor3, 1 / scale_factor3),
                                            interpolation=InterpolationMode.BILINEAR)

        # 将变换后的图像沿批次维度拼接
        img_ensemble = torch.cat((img_randomAffineScale1, img_randomAffineScale2, img_randomAffineScale3), dim=0)
        # 将拼接后的图像传递给网络
        outputs = self.net(img_ensemble)
        output_list = []
        # 解码网络的输出
        for k in range(3):
            output_list.append(self.yolo_decodes[k](outputs[k]))
        # 将解码后的输出沿第二维度拼接
        output = torch.cat(output_list, 1)  # batch_size×nx6 (xywh，obj_conf, cls_conf)
        # 通过对象置信度和类别置信度来计算分数
        obj_cls_conf = output[:, :, 4] + output[:, :, 5]
        # 获取每个批次的最大分数
        scores = torch.max(obj_cls_conf, dim=1).values
        # 为分数创建一个one-hot张量
        one_hot_output = torch.FloatTensor(scores.size()[-1]).zero_().cuda()
        one_hot_output[:] = 1
        # 清零网络的梯度
        self.net.zero_grad()
        # 进行反反向传播以计算梯度
        scores.backward(gradient=one_hot_output, retain_graph=retain_graph)

        '''-------------------multi-scale attention calculating----------------------'''
        # 获取原始图像的批量大小
        batch_size = img_origin.shape[0]

        # 初始化 Grad-CAM 缩放列表、原始 CAM 列表和归一化 CAM 列表
        grad_cam_resize_list = list()
        cam_origin_list = list()
        cam_origin_normalization_list = list()

        for i in range(len(self.gradient)):
            # 确保特征图的数量与梯度的数量匹配
            assert len(self.feature) == len(self.gradient) or len(self.feature) == 2 * len(self.gradient) \
                   or len(self.feature) == 3 * len(self.gradient) or len(self.feature) == 4 * len(
                self.gradient), 'Error! '
            # cam_weight = torch.mean(self.gradient[len(self.gradient) - i - 1].clone().detach(), dim=[2, 3],
            #                        keepdim=True) - torch.tensor([1]).cuda()
            # 获取设备信息
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            # 计算权重
            cam_weight = torch.mean(self.gradient[len(self.gradient) - i - 1].clone().detach(), dim=[2, 3],
                                    keepdim=True).to(device) - torch.tensor([1], device=device)
            # 计算原始 CAM
            origin_cam = torch.sum((cam_weight * self.feature[i]), dim=1)
            # 应用 ReLU 激活函数，并扩展维度生成正激活特征图
            relu_activated_map = torch.relu(origin_cam).unsqueeze(1)
            # 获取最小值
            relu_activated_map_min = torch.min(torch.min(relu_activated_map.clone().detach(), dim=3).values, dim=2).values
            relu_activated_map_min = relu_activated_map_min.unsqueeze(-1)
            relu_activated_map_min = relu_activated_map_min.expand(-1, -1, relu_activated_map.size(2))
            relu_activated_map_min = relu_activated_map_min.unsqueeze(-1)
            relu_activated_map_min = relu_activated_map_min.expand(-1, -1, -1, relu_activated_map.size(3))
            # 获取最大值
            relu_activated_map_max = torch.max(torch.max(relu_activated_map.clone().detach(), dim=3).values, dim=2).values
            relu_activated_map_max = relu_activated_map_max.unsqueeze(-1)
            relu_activated_map_max = relu_activated_map_max.expand(-1, -1, relu_activated_map.size(2))
            relu_activated_map_max = relu_activated_map_max.unsqueeze(-1)
            relu_activated_map_max = relu_activated_map_max.expand(-1, -1, -1, relu_activated_map.size(3))

            # 归一化 Grad-CAM
            grad_cam = (relu_activated_map - relu_activated_map_min) / (relu_activated_map_max - relu_activated_map_min)
            # save_image(grad_cam[0,:,:,:].unsqueeze(0).cpu().detach(), 'TotalImg_120.png')

            # 获取原始 CAM 和归一化后的 CAM
            cam_origin = origin_cam
            cam_origin_normalization = grad_cam
            # Normalization
            # 定义变换
            transform = transforms.Compose([transforms.Resize(size=(self.ori_shape[1], self.ori_shape[0]))])

            # 缩放 Grad-CAM
            grad_cam_resize = transform(grad_cam)

            # 计算每个批次的数量
            num = int(img_ensemble.shape[0] / batch_size)

            # ----------------inverse transformation on the attention-------------------#

            # 初始化掩码 mask0, mask1, mask2 在循环外部定义
            mask0 = None
            mask1 = None
            mask2 = None
            # 循环处理每个批次
            for j in range(num):
                mask_tmp = grad_cam_resize[batch_size * j:batch_size * (j + 1)]
                if j == 0:  #
                    # mask0=mask_tmp
                    mask_tmp1, _, _ = inverse_randomScale1(mask_tmp)
                    mask_tmp2, _, _ = inverse_randomAffine1(mask_tmp1)
                    mask0 = mask_tmp2
                elif j == 1:
                    mask_tmp1, _, _ = inverse_randomScale2(mask_tmp)
                    mask_tmp2, _, _ = inverse_randomAffine2(mask_tmp1)
                    mask1 = Hflip(mask_tmp2)

                elif j == 2:
                    mask_tmp1, _, _ = inverse_randomScale3(mask_tmp)
                    mask_tmp2, _, _ = inverse_randomAffine3(mask_tmp1)
                    mask2 = mask_tmp2

                # elif ss==3:
                #     mask_tmp1,_,_=inv_randomScale_b4(mask_tmp)
                #     mask_tmp2,_,_=inv_randomAffine_b4(mask_tmp1) 
                #     mask3=Hflip(mask_tmp2)

                else:
                    raise ValueError("Error.")
            # 初始化融合后的 Grad-CAM
            grad_cam_resize_ensem = None
            # 融合每个批次的结果
            for ll in range(batch_size):
                final_scores = 1. / num
                grad_cam_resize_ensem_tmp = \
                    mask0[ll].unsqueeze(0) * final_scores + \
                    mask1[ll].unsqueeze(0) * final_scores + \
                    mask2[ll].unsqueeze(0) * final_scores
                # + mask3[ll].unsqueeze(0)*final_scores
                if ll == 0:
                    grad_cam_resize_ensem = grad_cam_resize_ensem_tmp
                else:
                    grad_cam_resize_ensem = torch.cat((grad_cam_resize_ensem, grad_cam_resize_ensem_tmp), dim=0)
            # 添加到列表
            grad_cam_resize_list.append(grad_cam_resize_ensem)
            cam_origin_list.append(cam_origin[:batch_size])
            cam_origin_normalization_list.append(cam_origin_normalization[:batch_size])

        # 获取最后一个 Grad-CAM 的热力图
        mm = grad_cam_resize_list[1][-1, 0, :, :].cpu().detach()
        nn = np.uint8(mm * 255)
        heatmap = cv2.applyColorMap(nn, cv2.COLORMAP_JET)
        
        # 获取输入图像
        test_img = inputs['image'].clone()[-1, :, :, :].cpu().detach()
        
        # 将图像转换为 CV2 格式  [0,1]->[0,255]，CHW->HWC，->cv2
        test_img = test_img.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).type(torch.uint8).numpy()
        
        # 合成图像
        superimposed_img = heatmap * 0.6 + test_img * 0.4
        
        # 保存图像
        cv2.imwrite('./attention_map_sample.jpg', superimposed_img)  # layer5
        
        # 返回结果
        # grad_cam_resize_list：包含经过逆变换和归一化后的Grad-CAM特征图。它是通过对网络中不同层次的梯度进行处理后得到的，表示网络对输入图像不同区域的关注程度。
        # 这个列表中的每一项都是一个经过特定尺寸调整后的Grad-CAM特征图，经过反向仿射变换后，恢复到输入图像的原始尺度。

        # cam_origin_list：包含原始的 Grad CAM特征图，即在反向仿射变换和归一化之前的类激活映射。这些特征图用于表示网络某层特征对预测结果的贡献。.
        # cam_origin_normalization_list：包含归一化后的Grad-CAM特征图，归一化的目的是将特征图的值范围调整到[0, 1]，方便进行可视化或进一步处理。
        return grad_cam_resize_list, cam_origin_list, cam_origin_normalization_list
