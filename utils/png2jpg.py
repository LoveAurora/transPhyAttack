import os
from PIL import Image


def convert_png_to_jpg(input_folder):
    for filename in os.listdir(input_folder):
        if filename.endswith('.png'):
            # 构造文件路径
            png_path = os.path.join(input_folder, filename)
            jpg_path = os.path.join(input_folder, filename[:-4] + '.jpg')

            # 打开PNG图像并转换为JPG格式
            with Image.open(png_path) as img:
                rgb_img = img.convert('RGB')
                rgb_img.save(jpg_path, 'JPEG')

            # 删除原来的PNG文件
            os.remove(png_path)
            print(f"Converted {png_path} to {jpg_path} and deleted the original PNG file.")


# 文件夹路径
input_folder = './output_images_pth'  # 替换为你的文件夹路径
# 执行转换和删除操作
convert_png_to_jpg(input_folder)
print("All PNG images have been converted to JPG and original PNG files have been deleted.")

