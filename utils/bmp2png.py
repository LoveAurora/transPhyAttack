from PIL import Image
import os


def convert_bmp_folder_to_png(bmp_folder_path, png_folder_path):
    # 确保输出文件夹存在
    if not os.path.exists(png_folder_path):
        os.makedirs(png_folder_path)

    # 遍历文件夹中的所有文件
    for filename in os.listdir(bmp_folder_path):
        if filename.lower().endswith(".bmp"):
            bmp_file_path = os.path.join(bmp_folder_path, filename)
            png_filename = os.path.splitext(filename)[0] + ".png"
            png_file_path = os.path.join(png_folder_path, png_filename)

            try:
                # 打开BMP文件
                with Image.open(bmp_file_path) as img:
                    # 转换模式，确保图片是RGB模式
                    img = img.convert('RGB')
                    # 保存为PNG
                    img.save(png_file_path, 'PNG')
                    print(f"Converted {bmp_file_path} to {png_file_path}")
            except IOError as e:
                print(f"Error converting {bmp_file_path} to PNG: {e}")


# 使用示例
bmp_folder_path = '../bmw_normal'  # 替换为包含BMP文件的文件夹路径
png_folder_path = '../bmw_normal'  # 替换为你想保存PNG文件的文件夹路径


convert_bmp_folder_to_png(bmp_folder_path, png_folder_path)


def delete_bmp_files(folder_path):
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".bmp"):
            file_path = os.path.join(folder_path, filename)
            try:
                # 删除文件
                os.remove(file_path)
                print(f"Deleted {file_path}")
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")


# 使用示例
folder_path = '../bmw_normal'  # 替换为包含BMP文件的文件夹路径

delete_bmp_files(folder_path)
