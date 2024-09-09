import torch
import torch.nn as nn
import numpy as np

from PIL import Image
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import transforms


class NPSCalculator(nn.Module):
    """NMSCalculator: calculates the non-printability score of a patch.

    Module providing the functionality necessary to calculate the non-printability score (NMS) of an adversarial patch.

    """

    def __init__(self, printability_file, patch_size):
        """
          初始化 NPSCalculator 类。

          参数:
              printability_file (str): 打印能力文件的路径。
              patch_size (int): 用于计算的补丁大小。

          属性:
              printability_array (nn.Parameter): 打印能力数组，作为不可训练的参数。
          """
        # 调用父类的初始化方法
        super(NPSCalculator, self).__init__()

        # 从打印能力文件中获取打印能力数组，并将其转换为不可训练的参数
        self.printability_array = nn.Parameter(
            self.get_printability_array(printability_file, patch_size),
            requires_grad=False
        )

    def forward(self, adv_patch):
        """
        计算输入补丁与打印能力数组之间的颜色距离，并计算 NPS (No Print Score)。

        参数:
            adv_patch (torch.Tensor): 输入的补丁张量。

        返回:
            torch.Tensor: NPS 分数。
        """

        # 计算输入补丁与打印能力数组之间的欧氏距离
        # 首先计算颜色差的平方
        color_diff = (adv_patch - self.printability_array + 0.000001)
        color_diff_squared = color_diff ** 2

        # 对每个像素的颜色差求和
        color_dist = torch.sum(color_diff_squared, dim=4) + 0.000001

        # 计算欧氏距离
        color_dist = torch.sqrt(color_dist)

        # 只保留最小的距离
        color_dist_min = torch.min(color_dist, dim=0)[0]

        # 计算 NPS 分数，通过求和所有像素的最小距离
        nps_score = torch.sum(color_dist_min, dim=0)
        nps_score = torch.sum(nps_score, dim=0)
        nps_score = torch.sum(nps_score, dim=0)

        # 归一化 NPS 分数
        nps_score_normalized = nps_score / torch.numel(adv_patch)

        return nps_score_normalized

    @staticmethod
    def get_printability_array(printability_file, size):
        printability_list = []

        # read in printability triplets and put them in a list
        with open(printability_file) as f:
            for line in f:
                printability_list.append(line.split(","))

        printability_array = []
        for printability_triplet in printability_list:
            printability_imgs = []
            red, green, blue = printability_triplet
            printability_imgs.append(np.full((size[0], size[1], size[2]), red))
            printability_imgs.append(np.full((size[0], size[1], size[2]), green))
            printability_imgs.append(np.full((size[0], size[1], size[2]), blue))
            printability_array.append(printability_imgs)
        printability_array = np.asarray(printability_array)  #
        printability_array = np.float32(printability_array)
        pa = torch.from_numpy(printability_array).permute(0, 2, 3, 4, 1)
        return pa


class TotalVariation_3d(nn.Module):
    """TotalVariation: calculates the total variation of a patch.

    """

    def __init__(self, mesh, target_face_id):
        super(TotalVariation_3d, self).__init__()

        # https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/structures/meshes.html?highlight=faces_packed_to_edges_packed#
        # Map from packed faces to packed edges. This represents the index of
        # the edge opposite the vertex for each vertex in the face. E.g.
        #
        #         v0
        #         /\
        #        /  \
        #    e1 /    \ e2
        #      /      \
        #     /________\
        #   v2    e0   v1
        #
        # Face (v0, v1, v2) => Edges (e0, e1, e2)
        # 
        # Step 0: get all info of mesh[0]
        FtoE_id = mesh[0].faces_packed_to_edges_packed().cpu()
        EtoV_id = mesh[0].edges_packed().cpu()
        V = mesh[0].verts_packed()
        num_of_edges = EtoV_id.shape[0]
        num_of_target_faces = len(target_face_id)
        # Step 1: Construct (E_n, 2) tensor as opposite face indexing
        EtoF_idx1 = -1 * torch.ones((num_of_edges), dtype=torch.long)
        EtoF_idx2 = -1 * torch.ones((num_of_edges), dtype=torch.long)

        for i in range(num_of_target_faces):
            for each in FtoE_id[target_face_id[i]]:
                if EtoF_idx1[each] == -1:
                    EtoF_idx1[each] = i
                else:
                    EtoF_idx2[each] = i
                    # remove all edges that does not belong to
        valid_id = ~((EtoF_idx1 == -1) | (EtoF_idx2 == -1))
        EtoF_idx = torch.stack((EtoF_idx1[valid_id], EtoF_idx2[valid_id]), dim=1)
        self.face_to_edges_idx = EtoF_idx.cuda()  #
        # Step 2: Compute edge length
        valid_edge = EtoV_id[valid_id, :]  #
        self.edge_len = torch.norm(V[valid_edge[:, 0], :] - V[valid_edge[:, 1], :], dim=1).cuda()

    def forward(self, adv_patch):

        f1 = adv_patch[self.face_to_edges_idx[:, 0], :, :, :]
        f2 = adv_patch[self.face_to_edges_idx[:, 1], :, :, :]
        tv = torch.sum(self.edge_len[:, None, None, None] * torch.abs(f1 - f2))
        return tv / adv_patch.shape[0]


def bbox_iou(bbox_a, bbox_b):
    if bbox_a.shape[1] != 4 or bbox_b.shape[1] != 4:
        print(bbox_a, bbox_b)
        raise IndexError
    tl = np.maximum(bbox_a[:, None, :2], bbox_b[:, :2])  #
    # bottom right
    br = np.minimum(bbox_a[:, None, 2:], bbox_b[:, 2:])
    area_i = np.prod(br - tl, axis=2) * (tl < br).all(axis=2)
    area_a = np.prod(bbox_a[:, 2:] - bbox_a[:, :2], axis=1)
    area_b = np.prod(bbox_b[:, 2:] - bbox_b[:, :2], axis=1)
    return area_i / (area_a[:, None] + area_b - area_i)


def bbox_iou_V2(box1, box2, x1y1x2y2=True):
    """
        calculating IOU
    """
    if not x1y1x2y2:
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * \
                 torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)

    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou


class MaxProbExtractor(nn.Module):
    """MaxProbExtractor: extracts max class probability for class from YOLO output_images.

    Module providing the functionality necessary to extract the max class probability for one class from YOLO output_images.

    """

    def __init__(self, cls_id, num_cls, config):
        super(MaxProbExtractor, self).__init__()
        self.cls_id = cls_id
        self.num_cls = num_cls  # 80
        self.config = config

    def forward(self, YOLOoutput, conf_thres, file_name_BS):

        output_objectness = YOLOoutput[:, :, 4]
        normal_confs = YOLOoutput[:, :, 5:]
        confs_for_class = normal_confs[:, :, self.cls_id]

        loss_target = lambda obj, cls: obj + cls
        confs_if_object = loss_target(output_objectness, confs_for_class)

        '''get IOU'''
        prediction = YOLOoutput
        box_corner = prediction.new(prediction.shape)
        box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
        box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
        box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
        box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
        prediction[:, :, :4] = box_corner[:, :, :4]
        max_ious = 0
        for image_i, image_pred in enumerate(prediction):

            '''getting GT'''
            GT_path = "TrainingGT/ground-truth/" + file_name_BS[image_i].split('\\')[-1].split('.npz')[0] + ".txt"
            lines = file_lines_to_list(GT_path)
            for line in lines:
                try:
                    tmp_class_name, left, top, right, bottom = line.split()
                except:
                    line_split = line.split()
                    bottom = int(line_split[-1])
                    right = int(line_split[-2])
                    top = int(line_split[-3])
                    left = int(line_split[-4])
                GT_bbox = [int(left), int(top), int(right), int(bottom)]
            ious = bbox_iou_V2(torch.tensor(GT_bbox).cuda().unsqueeze(0), image_pred[0:])
            k_num = 20
            ious_topK = torch.topk(ious, k=k_num, dim=0, largest=True)
            topK_ious_value, topK_confs_value_indice = ious_topK[0], ious_topK[1]
            max_ious = max_ious + topK_ious_value.sum(dim=0) / k_num
        k_num = 20
        topK = torch.topk(confs_if_object, k=k_num, dim=1, largest=True)
        topK_confs_value, topK_confs_value_indice = topK[0], topK[1]
        max_conf = topK_confs_value.sum(dim=1) / k_num
        return max_conf, max_ious


"""
Convert the lines of a file to a list
"""


def file_lines_to_list(path):
    # open txt file lines to a list
    with open(path) as f:
        content = f.readlines()
    # remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content]
    return content
