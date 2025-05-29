import os
import numpy as np
from PIL import Image


def save_masks_from_npy(npy_file_path, output_directory):
    # 读取 .npy 文件
    masks_array = np.load(npy_file_path)

    # 确保输出目录存在，如果不存在则创建
    os.makedirs(output_directory, exist_ok=True)

    # 遍历数组中的每个图像，并保存为图像文件
    for i, mask in enumerate(masks_array):
        # 将图像数组转换为 PIL 图像对象
        img = Image.fromarray(mask.astype(np.uint8))  # 如果是灰度图像，可以直接转换为 uint8 类型

        # 定义保存路径，使用序号命名图像
        save_path = os.path.join(output_directory, f"mask_{i + 1}.png")

        # 保存图像
        img.save(save_path)
        print(f"Saved {save_path}")


# 使用示例
npy_file_path = '/media/daxu/diskd/Users/Administrator/Pycharm/DDNM/exp/inp_masks/mask.npy'  # .npy 文件的路径
output_directory = '/media/daxu/diskd/Users/Administrator/Pycharm/DDNM/exp/inp_masks/masks_npy'  # 保存图像的目标目录

# 调用函数
save_masks_from_npy(npy_file_path, output_directory)
