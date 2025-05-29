import os
import numpy as np
import imageio
from PIL import Image


def load_masks_from_directory(directory_path, save_path):
    # 获取目录下所有图像文件（假设是 PNG 格式）
    mask_files = [f for f in os.listdir(directory_path) if f.endswith('.png')]

    # 按文件名排序（假设文件名为数字，如 000000.png, 000001.png, ..., 002992.png）
    mask_files.sort()

    # 读取所有图像并将它们保存到一个列表
    masks = []
    for mask_file in mask_files:
        # 完整的图像路径
        mask_path = os.path.join(directory_path, mask_file)

        # 读取图像（使用 imageio 或 PIL 都可以）
        img = imageio.imread(mask_path)  # 或者使用 PIL: img = Image.open(mask_path)

        # 假设所有图像大小相同，直接添加到列表中
        masks.append(img)

    # 将列表转换为 NumPy 数组
    masks_array = np.array(masks)

    # 保存为指定路径的 .npy 文件
    np.save(save_path, masks_array)

    print(f"Successfully saved {len(masks)} masks to {save_path}")
    return masks_array


# 使用示例
directory_path = '/media/daxu/diskd/Datasets/CelebA-HQ/celeba-hq/masks_256_large'  # 掩码图像所在目录
save_path = '/media/daxu/diskd/Users/Administrator/Pycharm/DDNM/exp/datasets/masks/masks_large.npy'  # 指定保存路径

# 调用函数
masks_array = load_masks_from_directory(directory_path, save_path)
