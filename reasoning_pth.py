import time
from PIL import Image
import torch.nn as nn
import colorsys
import torchvision
import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator
# 假设这些模块已经在YoloV3库中定义好
from YoloV3.nets.yolo3 import YoloBody  # 网络主体
from YoloV3.utils.utils import DecodeBox  # 解码器
from YoloV3 import attention  # 注意力机制模块


def rescale_boxes(boxes, current_dim, original_shape):
    """
    Rescales bounding boxes to the original shape
    """
    orig_h, orig_w = original_shape

    # The amount of padding that was added
    pad_x = max(orig_h - orig_w, 0) * (current_dim / max(original_shape))
    pad_y = max(orig_w - orig_h, 0) * (current_dim / max(original_shape))

    # Image height and width after padding is removed
    unpad_h = current_dim - pad_y
    unpad_w = current_dim - pad_x

    # Rescale bounding boxes to dimension of original image
    boxes[:, 0] = ((boxes[:, 0] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 1] = ((boxes[:, 1] - pad_y // 2) / unpad_h) * orig_h
    boxes[:, 2] = ((boxes[:, 2] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 3] = ((boxes[:, 3] - pad_y // 2) / unpad_h) * orig_h
    return boxes


def xywh2xyxy(x):
    y = x.new(x.shape)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None):
    """Performs Non-Maximum Suppression (NMS) on inference results
    Returns:
         detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
    """

    nc = prediction.shape[2] - 5  # number of classes

    # Settings
    # (pixels) minimum and maximum box width and height
    max_wh = 4096
    max_det = 300  # maximum number of detections per image
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 1.0  # seconds to quit after
    multi_label = nc > 1  # multiple labels per box (adds 0.5ms/img_normal)

    t = time.time()
    output = [torch.zeros((0, 6), device="cpu")] * prediction.shape[0]

    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[x[..., 4] > conf_thres]  # confidence

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            # sort by confidence
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]

        # Batched NMS
        c = x[:, 5:6] * max_wh  # classes
        # boxes (offset by class), scores
        boxes, scores = x[:, :4] + c, x[:, 4]
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]

        def to_cpu(tensor):
            return tensor.detach().cpu()

        output[xi] = to_cpu(x[i])

        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded

    return output


def draw_and_save_output_image(image_path, detections, img_size, output_path, classes):
    """Draws detections in output_images image and stores this.

    :param image_path: Path to input image
    :type image_path: str
    :param detections: List of detections on image
    :type detections: [Tensor]
    :param img_size: Size of each image dimension for yolo
    :type img_size: int
    :param output_path: Path of output_images directory
    :type output_path: str
    :param classes: List of class names
    :type classes: [str]
    """
    # Create plot
    # Create plot
    img = np.array(Image.open(image_path))
    plt.figure()
    fig, ax = plt.subplots(1)
    ax.imshow(img)
    # Rescale boxes to original image
    detections = rescale_boxes(detections, img_size, img.shape[:2])
    unique_labels = detections[:, -1].cpu().unique()
    n_cls_preds = len(unique_labels)
    # Bounding-box colors
    cmap = plt.get_cmap("tab20b")
    colors = [cmap(i) for i in np.linspace(0, 1, n_cls_preds)]
    bbox_colors = random.sample(colors, n_cls_preds)
    for x1, y1, x2, y2, conf, cls_pred in detections:
        print(f"\t+ Label: {classes[int(cls_pred)]} | Confidence: {conf.item():0.4f}")

        box_w = x2 - x1
        box_h = y2 - y1

        color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
        # Create a Rectangle patch
        bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")
        # Add the bbox to the plot
        ax.add_patch(bbox)
        # Add label
        plt.text(
            x1,
            y1,
            s=f"{classes[int(cls_pred)]}: {conf:.2f}",
            color="white",
            verticalalignment="top",
            bbox={"color": color, "pad": 0})

    # Save generated image with detections
    plt.axis("off")
    plt.gca().xaxis.set_major_locator(NullLocator())
    plt.gca().yaxis.set_major_locator(NullLocator())
    output_path = os.path.join(output_path, os.path.basename(image_path))  # Use original filename
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0.0)
    plt.close()


class YOLO(nn.Module):
    def __init__(self, config):
        super(YOLO, self).__init__()
        self.config = config
        self.model_path = config.get("model_path", '')
        self.anchors_path = config.get("anchors_path", 'YoloV3/utils/yolo_anchors.txt')
        self.classes_path = config.get("classes_path", 'YoloV3/utils/CARLA_classes.txt')
        self.model_image_size = config.get("model_image_size", (608, 608, 1))
        self.confidence = config.get("confidence", 0.25)
        self.iou = config.get("iou", 0.5)
        self.cuda = config.get("cuda", True)
        self.letterbox_image = config.get("letterbox_image", False)

        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.num_classes = len(self.class_names)
        self.net = YoloBody(self.anchors, self.num_classes)

        self._load_model_weights()
        self._setup_device()
        self._prepare_decodes()
        self._prepare_colors()
        self.multi_attention = attention.Attention(self.net, ori_shape=self.model_image_size,
                                                   final_shape=self.model_image_size,
                                                   yolo_decodes=self.yolo_decodes,
                                                   num_classes=self.num_classes,
                                                   conf_thres=self.confidence,
                                                   nms_thres=self.iou)

    def _load_model_weights(self):
        # 加载预训练的模型权重
        print('Loading weights into state dict...')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_path = self.config.get("weightfile", self.model_path)
        state_dict = torch.load(model_path, map_location=device)
        self.net.load_state_dict(state_dict)
        self.net = self.net.eval()
        print('Model weights loaded.')

    def _setup_device(self):
        # 设置设备
        if self.cuda:
            self.net = self.net.cuda()

    def _prepare_decodes(self):
        # 准备解码器
        self.yolo_decodes = []
        for i in range(3):
            self.yolo_decodes.append(
                DecodeBox(self.anchors[i], self.num_classes, (self.model_image_size[1], self.model_image_size[0])))

    def _prepare_colors(self):
        # 准备颜色映射
        hsv_tuples = [(x / len(self.class_names), 1., 1.) for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape([-1, 3, 2])[::-1, :, :]

    def forward(self, x):
        outputs = self.net(x)
        output_list = []
        for i in range(3):
            output_list.append(self.yolo_decodes[i](outputs[i]))
        output = torch.cat(output_list, 1)
        return output


# 示例使用
config = {
    "model_path": r'',  # YoloV3\checkpoint\Epoch300-Total_Loss1.9337-Val_Loss1.9373.pth
    "anchors_path": r'YoloV3/utils/yolo_anchors.txt',
    "classes_path": r'YoloV3/utils/CARLA_classes.txt',
    "model_image_size": (608, 608, 3),  # 确保图像尺寸的形状正确
    "confidence": 0.25,
    "iou": 0.5,
    "cuda": True,
    "letterbox_image": False,
    "weightfile": './YoloV3/checkpoint/Epoch300-Total_Loss1.9337-Val_Loss1.9373.pth'
}

yolo_model = YOLO(config)

# 文件夹路径

configs = {
    'normal': {
        'input_folder': './img_normal',
        # 'output_folder': 'output_images_normal_pth_ratio',
        'output_folder': 'output_images_pth',
    },
    'patch': {
        'input_folder': './img_patch',
        # 'output_folder': 'output_images_patch_pth_ratio',
        'output_folder': 'output_images_pth',
    }
}
# 改这个变量名字切换不同的配置
config_key = 'normal'
total_imgs = len(os.listdir(configs[config_key]['input_folder']))
not_detected_imgs_filename = list()
detected_imgs = 0


def preprocess_image(imgs_path, input_size):
    # 打开图片并进行预处理
    img = Image.open(imgs_path).convert('RGB')
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(input_size),
        torchvision.transforms.ToTensor()
    ])
    img_tensor = transform(img).unsqueeze(0).cuda()
    return img_tensor


def process_image(imgs_path):
    input_size = (608, 608)
    img_tensor = preprocess_image(imgs_path, input_size)

    output = yolo_model(img_tensor)
    nms_output = non_max_suppression(output, conf_thres=0.25, iou_thres=0.45, classes=None)
    if len(nms_output[0]) > 0:
        global detected_imgs
        detected_imgs += 1
    # 提取未被检测到的图片的文件名
    # else:
    #     global not_detected_imgs_filename
    #     not_detected_imgs_filename.append(imgs_path)
    # print(f"Found {len(nms_output[0])} objects in {imgs_path}")
    # print(f"Found {len(nms_output[0])} objects in {imgs_path}")
    # 保存图片
    # draw_and_save_output_image(imgs_path, nms_output[0], 608, "output_images_pth", ['car'])
    plt.close('all')  # 关闭所有图形以释放内存


# 如果输出文件夹不存在，则创建它
if not os.path.exists(configs[config_key]['output_folder']):
    os.makedirs(configs[config_key]['output_folder'])

# 遍历文件夹中的所有图片文件
for filename in os.listdir(configs[config_key]['input_folder']):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        img_path = os.path.join(configs[config_key]['input_folder'], filename)
        process_image(img_path)
        print(f"Processed {img_path}")

# 打开一个文件进行写入（如果文件不存在则创建）
# 把没有被检测到的图片的文件名写入文件
# with open('detected_images_patch_pth.txt', 'w') as file:
#     for filename in not_detected_imgs_filename:
#         # 写入文件名，并在每个文件名后添加换行符
#         file.write(filename + '\n')

success_rate = 0
if config_key == 'normal':
    success_rate = detected_imgs / total_imgs
elif config_key == 'patch':
    success_rate = 1 - detected_imgs / total_imgs

print(f"Success rate {success_rate} ")
