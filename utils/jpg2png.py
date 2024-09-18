import os
from PIL import Image


def convert_jpg_to_png(input_folder):
    for filename in os.listdir(input_folder):
        if filename.lower().endswith('.jpg'):
            # 构造文件路径
            jpg_path = os.path.join(input_folder, filename)
            png_path = os.path.join(input_folder, filename[:-4] + '.png')

            # 打开PNG图像并转换为JPG格式
            with Image.open(jpg_path) as img:
                rgb_img = img.convert('RGB')
                rgb_img.save(png_path, 'PNG')

            # 删除原来的PNG文件
            os.remove(jpg_path)
            print(f"Converted {jpg_path} to {png_path} and deleted the original PNG file.")


# 文件夹路径
input_folder = '../bmw_patch'  # 替换为你的文件夹路径
# 执行转换和删除操作
convert_jpg_to_png(input_folder)
print("All PNG images have been converted to JPG and original PNG files have been deleted.")
