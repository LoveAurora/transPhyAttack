import numpy as np
import os
import cv2


def extract_images_from_npz(npz_file, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    data = np.load(npz_file)

    for key in data.files:
        if key != 'img':
            continue  # 只处理 'img_normal' 键的数据

        array = data[key]
        print(f"Processing key: {key} with shape: {array.shape}")

        # 处理图像数据
        if array.ndim == 4:  # (N, H, W, C)
            num_images = array.shape[0]
            for i in range(num_images):
                image = array[i]
                output_path = os.path.join(output_folder,
                                           f"{os.path.splitext(os.path.basename(npz_file))[0]}_{key}_{i}.png")
                cv2.imwrite(output_path, image)
                print(f"Saved image to: {output_path}")

        elif array.ndim == 3:  # (H, W, C)
            output_path = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(npz_file))[0]}_{key}.png")
            cv2.imwrite(output_path, array)
            print(f"Saved image to: {output_path}")

        else:
            print(f"Unexpected data shape for key {key} with shape {array.shape}")


def process_npz_files(input_folder, output_folder):
    npz_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith('.npz')]

    for npz_file in npz_files:
        print(f"Processing file: {npz_file}")
        extract_images_from_npz(npz_file, output_folder)


# 示例用法
input_folder = './dataset/testset'  # 替换为包含 .npz 文件的文件夹路径
output_folder = './img_normal'  # 替换为保存图像的文件夹路径
process_npz_files(input_folder, output_folder)
