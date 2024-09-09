import time
from PIL import Image
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator
import torch
import onnxruntime as ort
import torchvision


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
    max_wh = 4096
    max_det = 300  # maximum number of detections per image
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 1.0  # seconds to quit after
    multi_label = nc > 1  # multiple labels per box (adds 0.5ms/img_normal)

    t = time.time()
    output = [torch.zeros((0, 6), device="cpu")] * prediction.shape[0]

    for xi, x in enumerate(prediction):  # image index, image inference
        x = torch.tensor(x)
        # Apply constraints
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
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]

        # Batched NMS
        c = x[:, 5:6] * max_wh  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]

        output[xi] = x[i].cpu()

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
    file_name = os.path.basename(image_path).split(".")[0]
    output_path = os.path.join(output_path, f"{file_name}.png")
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0.0)
    plt.close()


def preprocess_image(imgs_path, input_size):
    # 打开图片并进行预处理
    img = Image.open(imgs_path).convert('RGB')
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(input_size),
        torchvision.transforms.ToTensor()
    ])
    img_tensor = transform(img).unsqueeze(0).numpy()
    return img, img_tensor


def postprocess_output(detections, img_shape, img_size):
    # 将检测框调整到原图尺寸
    boxes = detections[0]
    rescaled_boxes = rescale_boxes(boxes, img_size, img_shape)
    return rescaled_boxes


# 载入 ONNX 模型
onnx_model_path = 'yolov3.onnx'
ort_session = ort.InferenceSession(onnx_model_path)

configs = {
    'normal': {
        'input_folder': './img_normal',
        'output_folder': 'output_images_normal_pth_ratio',
    },
    'patch': {
        'input_folder': './img_patch',
        'output_folder': 'output_images_patch_pth_ratio',
    }
}
# 改这个变量名字切换不同的配置
config_key = 'normal'
# config_key = 'patch'
total_imgs = len(os.listdir(configs[config_key]['input_folder']))
not_detected_imgs_filename = list()
detected_imgs = 0


def process_image(imgs_path):
    input_size = (608, 608)
    img, img_tensor = preprocess_image(imgs_path, input_size)

    # ONNX 模型推理
    outputs = ort_session.run(None, {ort_session.get_inputs()[0].name: img_tensor})

    # 将输出转换为Tensor并进行后处理
    output_tensor = torch.tensor(outputs[0]).clone().detach()  # 避免警告
    nms_output = non_max_suppression(output_tensor, conf_thres=0.25, iou_thres=0.45, classes=None)
    # 后处理 NMS 输出
    # 因为是一张图片，所以只取第一个元素
    if len(nms_output[0]) > 0:
        global detected_imgs
        detected_imgs += 1
    # 提取未被检测到的图片的文件名
    # else:
    #     global not_detected_imgs_filename
    #     not_detected_imgs_filename.append(imgs_path)
    # 保存图片
    # draw_and_save_output_image(imgs_path, nms_output[0], 608, "output_images_onnx", ['car'])
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
# with open('detected_images_patch_onnx.txt', 'w') as file:
#     for filename in detected_imgs_filename:
#         # 写入文件名，并在每个文件名后添加换行符
#         file.write(filename + '\n')

success_rate = 0
if config_key == 'normal':
    success_rate = detected_imgs / total_imgs
elif config_key == 'patch':
    success_rate = 1 - detected_imgs / total_imgs

print(f"Success rate {success_rate} ")
