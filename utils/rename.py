import os


def rename_images(input_folder):
    # 获取文件夹中所有png和jpg文件的列表
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg'))]
    # 按文件名顺序对文件列表进行排序
    image_files.sort()

    # 初始化文件索引
    index = 1

    # 按顺序重命名文件
    for filename in image_files:
        # 构造原始文件路径
        old_path = os.path.join(input_folder, filename)
        # 获取文件扩展名
        ext = os.path.splitext(filename)[1]
        # 构造新文件名（例如：1.png, 2.jpg, ...）
        new_filename = f"{index}{ext}"
        # 构造新文件路径
        new_path = os.path.join(input_folder, new_filename)

        # 重命名文件
        os.rename(old_path, new_path)
        print(f"Renamed {old_path} to {new_path}")
        # 更新文件索引
        index += 1


# 文件夹路径
input_folder = '../bmw_normal'  # 替换为你的文件夹路径
# 执行重命名操作
rename_images(input_folder)
print("All PNG and JPG files have been renamed.")
