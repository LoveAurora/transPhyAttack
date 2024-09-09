import os

import os


def find_and_remove_unmatched_images(folder_path):
    # 获取文件夹中所有的文件名
    files = os.listdir(folder_path)
    print(f"Found {len(files)} files in folder: {folder_path}")

    # 创建集合存储基础文件名和带_img的文件名
    base_filenames = set()
    img_filenames = set()

    # 遍历所有文件
    for file in files:
        if file.endswith('.jpg'):
            if '_img' in file:
                base_name = file.replace('_img.jpg', '.jpg')
                img_filenames.add(file)
            else:
                base_name = file
            base_filenames.add(base_name)

    # 打印基础文件名和带_img文件名
    # print("Base filenames:")
    # for fname in base_filenames:
    #     print(f"  {fname}")
    #
    # print("Image filenames:")
    # for fname in img_filenames:
    #     print(f"  {fname}")

    # 查找并删除没有对应文件的文件
    unmatched_files = []
    # 删除没有匹配 `_img` 文件的基础文件
    for base_file in base_filenames:
        img_file = base_file.replace('.jpg', '_img.jpg')
        if img_file not in img_filenames:
            unmatched_files.append(base_file)

    # 删除没有匹配基础文件的 `_img` 文件
    for img_file in img_filenames:
        base_file = img_file.replace('_img.jpg', '.jpg')
        if base_file not in base_filenames:
            unmatched_files.append(img_file)

    # 删除所有没有匹配的文件
    for file in unmatched_files:
        file_path = os.path.join(folder_path, file)
        os.remove(file_path)
        print(f"Removed unmatched file: {file}")


# 指定文件夹路径
folder_path = './output_images_pth'

# 执行查找和删除操作
find_and_remove_unmatched_images(folder_path)
print("All unmatched images have been removed.")
