import os
import cv2
import numpy as np


def invert_mask_images(input_dir, output_dir):
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 遍历输入目录中的所有文件
    for file_name in os.listdir(input_dir):
        input_path = os.path.join(input_dir, file_name)

        # 检查文件是否为图像文件
        if os.path.isfile(input_path) and file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            # 读取图像（以灰度模式）
            mask = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print(f"无法读取文件: {file_name}")
                continue

            # 反转图像 (黑变白，白变黑)
            inverted_mask = cv2.bitwise_not(mask)

            # 保存到输出目录
            output_path = os.path.join(output_dir, file_name)
            cv2.imwrite(output_path, inverted_mask)

            print(f"已处理: {file_name} -> {output_path}")


# 输入和输出目录
input_directory = "/media/daxu/diskd/Datasets/CelebA-HQ/celeba-hq/masks_512_large"  # 替换为你的输入目录路径
output_directory = "/media/daxu/diskd/Datasets/CelebA-HQ/celeba-hq/masks_512_large_rev"  # 替换为你的输出目录路径

invert_mask_images(input_directory, output_directory)
