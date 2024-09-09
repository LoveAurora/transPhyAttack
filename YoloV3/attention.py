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

        img_ori = inputs['image']
        Hflip = transforms.RandomHorizontalFlip(p=1)

        '''第一个复合变换：平移->缩放'''
        # 平移
        randomAffine1 = myRandomAffine(degrees=(0, 0), translate=(0.1, 0.1))
        img_randomAffine1, translations_b1, _ = randomAffine1(img_ori.clone())
        # 逆平移
        inv_randomAffine_b1 = myRandomAffine(degrees=(0, 0), inv_translate=(-translations_b1[0], -translations_b1[1]))
        # 缩放
        randomAffine_scale1 = myRandomAffine(degrees=(0, 0), scale=(0.8, 1.2), interpolation=InterpolationMode.BILINEAR)
        img_randomAffineScale1, _, scale_factor_b1 = randomAffine_scale1(img_randomAffine1)
        # 逆缩放
        inv_randomScale_b1 = myRandomAffine(degrees=(0, 0), scale=(1 / scale_factor_b1, 1 / scale_factor_b1),
                                            interpolation=InterpolationMode.BILINEAR)

        '''第二个：水平翻转->平移->缩放'''
        # 水平翻转
        img_flip2 = Hflip(img_ori.clone())
        # 平移
        randomAffine2 = myRandomAffine(degrees=(0, 0), translate=(0.1, 0.1))
        img_randomAffine2, translations_b2, _ = randomAffine2(img_flip2)
        # 逆平移
        inv_randomAffine_b2 = myRandomAffine(degrees=(0, 0), inv_translate=(-translations_b2[0], -translations_b2[1]))
        # 缩放
        randomAffine_scale2 = myRandomAffine(degrees=(0, 0), scale=(0.8, 1.2), interpolation=InterpolationMode.BILINEAR)
        img_randomAffineScale2, _, scale_factor_b2 = randomAffine_scale2(img_randomAffine2)
        # 逆缩放
        inv_randomScale_b2 = myRandomAffine(degrees=(0, 0), scale=(1 / scale_factor_b2, 1 / scale_factor_b2),
                                            interpolation=InterpolationMode.BILINEAR)

        '''第三个：平移->缩放'''
        # 平移
        randomAffine3 = myRandomAffine(degrees=(0, 0), translate=(0.1, 0.1))
        img_randomAffine3, translations_b3, _ = randomAffine3(img_ori.clone())
        # 逆平移
        inv_randomAffine_b3 = myRandomAffine(degrees=(0, 0), inv_translate=(-translations_b3[0], -translations_b3[1]))
        # 缩放
        randomAffine_scale3 = myRandomAffine(degrees=(0, 0), scale=(0.8, 1.2), interpolation=InterpolationMode.BILINEAR)
        img_randomAffineScale3, _, scale_factor_b3 = randomAffine_scale3(img_randomAffine3)
        # 逆缩放
        inv_randomScale_b3 = myRandomAffine(degrees=(0, 0), scale=(1 / scale_factor_b3, 1 / scale_factor_b3),
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
        output = torch.cat(output_list, 1)  # BS×nx6 (xywh，obj_conf, cls_conf)
        # 通过对象置信度和类别置信度来计算分数
        bb = output[:, :, 4] + output[:, :, 5]
        # 获取每个批次的最大分数
        scores = torch.max(bb, dim=1).values
        # 为分数创建一个one-hot张量
        one_hot_output = torch.FloatTensor(scores.size()[-1]).zero_().cuda()
        one_hot_output[:] = 1
        # 清零网络的梯度
        self.net.zero_grad()
        # 进行反反向传播以计算梯度
        scores.backward(gradient=one_hot_output, retain_graph=retain_graph)

        '''-------------------multi-scale attention calculating----------------------'''
        BS = img_ori.shape[0]
        grad_cam_resize_list = list()
        cam_ori_list = list()
        cam_ori_norm_list = list()
        for kk in range(len(self.gradient)):
            assert len(self.feature) == len(self.gradient) or len(self.feature) == 2 * len(self.gradient) \
                   or len(self.feature) == 3 * len(self.gradient) or len(self.feature) == 4 * len(
                self.gradient), 'Error! '
            # cam_weight = torch.mean(self.gradient[len(self.gradient) - kk - 1].clone().detach(), dim=[2, 3],
            #                        keepdim=True) - torch.tensor([1]).cuda()
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            cam_weight = torch.mean(self.gradient[len(self.gradient) - kk - 1].clone().detach(), dim=[2, 3],
                                    keepdim=True).to(device) - torch.tensor([1], device=device)
            aa = torch.sum((cam_weight * self.feature[kk]), dim=1)
            bb = torch.relu(aa).unsqueeze(1)
            bb_min = torch.min(torch.min(bb.clone().detach(), dim=3).values, dim=2).values
            bb_min = bb_min.unsqueeze(-1)
            bb_min = bb_min.expand(-1, -1, bb.size(2))
            bb_min = bb_min.unsqueeze(-1)
            bb_min = bb_min.expand(-1, -1, -1, bb.size(3))
            bb_max = torch.max(torch.max(bb.clone().detach(), dim=3).values, dim=2).values
            bb_max = bb_max.unsqueeze(-1)
            bb_max = bb_max.expand(-1, -1, bb.size(2))
            bb_max = bb_max.unsqueeze(-1)
            bb_max = bb_max.expand(-1, -1, -1, bb.size(3))
            grad_cam = (bb - bb_min) / (bb_max - bb_min)
            # save_image(grad_cam[0,:,:,:].unsqueeze(0).cpu().detach(), 'TotalImg_120.png')
            cam_ori = aa
            cam_ori_norm = grad_cam
            transform = transforms.Compose([transforms.Resize(size=(self.ori_shape[1], self.ori_shape[0]))])
            grad_cam_resize = transform(grad_cam)

            num = int(img_ensemble.shape[0] / BS)  #

            # ----------------inverse transformation on the attention-------------------#
            # print(f"After randomAffine1 shape: {img_randomAffine1.shape}")
            # print(f"After randomAffineScale1 shape: {img_randomAffineScale1.shape}")
            #
            # print(f"After randomAffine1 shape: {img_randomAffine2.shape}")
            # print(f"After randomAffineScale1 shape: {img_randomAffineScale2.shape}")
            #
            # print(f"After randomAffine1 shape: {img_randomAffine3.shape}")
            # print(f"After randomAffineScale1 shape: {img_randomAffineScale3.shape}")

            # mask0, mask1, mask2 在循环外部定义
            mask0 = None
            mask1 = None
            mask2 = None
            for ss in range(num):
                mask_tmp = grad_cam_resize[BS * ss:BS * (ss + 1)]
                if ss == 0:  #
                    # mask0=mask_tmp
                    mask_tmp1, _, _ = inv_randomScale_b1(mask_tmp)
                    mask_tmp2, _, _ = inv_randomAffine_b1(mask_tmp1)
                    mask0 = mask_tmp2
                elif ss == 1:
                    mask_tmp1, _, _ = inv_randomScale_b2(mask_tmp)
                    mask_tmp2, _, _ = inv_randomAffine_b2(mask_tmp1)
                    mask1 = Hflip(mask_tmp2)

                elif ss == 2:
                    mask_tmp1, _, _ = inv_randomScale_b3(mask_tmp)
                    mask_tmp2, _, _ = inv_randomAffine_b3(mask_tmp1)
                    mask2 = mask_tmp2

                # elif ss==3:
                #     mask_tmp1,_,_=inv_randomScale_b4(mask_tmp)
                #     mask_tmp2,_,_=inv_randomAffine_b4(mask_tmp1) 
                #     mask3=Hflip(mask_tmp2)

                else:
                    raise ValueError("Error.")

            grad_cam_resize_ensem = None
            for ll in range(BS):
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
            grad_cam_resize_list.append(grad_cam_resize_ensem)
            cam_ori_list.append(cam_ori[:BS])
            cam_ori_norm_list.append(cam_ori_norm[:BS])

        mm = grad_cam_resize_list[1][-1, 0, :, :].cpu().detach()
        nn = np.uint8(mm * 255)
        heatmap = cv2.applyColorMap(nn, cv2.COLORMAP_JET)

        test_img = inputs['image'].clone()[-1, :, :, :].cpu().detach()
        # [0,1]->[0,255]，CHW->HWC，->cv2
        test_img = test_img.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).type(torch.uint8).numpy()
        superimposed_img = heatmap * 0.6 + test_img * 0.4
        cv2.imwrite('./attention_map_sample.jpg', superimposed_img)  # layer5
        return grad_cam_resize_list, cam_ori_list, cam_ori_norm_list
