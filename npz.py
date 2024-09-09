# import os
# import numpy as np
#
# # 设置要检查的目录
# directory = "./dataset/trainset"
#
# # 获取所有 .npz 文件的路径
# npz_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.npz')]
# print(f"发现 {len(npz_files)} 个 .npz 文件。")
# # 检查每个文件是否能正确加载并删除无法加载的文件
# for file in npz_files:
#     try:
#         np.load(file)
#     except Exception as e:
#         print(f"删除无法加载的文件: {file}")
#         os.remove(file)
#
# print("处理完成。")

import numpy as np

# 假设你的.npz文件名为'data.npz'
file_name = 'dataset/testset/Town10HD_Opt_P27_cam_dis10_x0.0y0.0z10.0p-90.0y180.0r0.0veh_yaw0.0.npz'

# 加载.npz文件
with np.load(file_name) as data:
    # 获取档案中的所有键（即数组的名称）
    keys = data.files

    # 遍历所有的键，并提取对应的数组
    for key in keys:
        # 提取数组
        array = data[key]
        print(f"{key}: {array}")  # 打印数组的名称和形状

        # 如果你想将提取的数组保存为.npy文件，可以这样做：
        # np.save(f"{key}.npy", array)
