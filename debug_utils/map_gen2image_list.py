import os
import pandas as pd
import shutil

# 读取 image_list.txt 文件
image_list_path = "/media/daxu/diskd/Datasets/celebahq/celebahq/deltas/image_list.txt"
image_list = pd.read_csv(image_list_path, delim_whitespace=True)

# 定义原始图像文件目录和目标保存目录
source_image_dir = "/media/daxu/diskd/Datasets/CelebA-HQ/CelebA-HQ-1024"  # 原始图像文件夹路径
target_image_dir = "/media/daxu/diskd/Datasets/CelebA-HQ/celebahq1024"  # 重命名后的文件存储目录

# 如果目标目录不存在，创建它
if not os.path.exists(target_image_dir):
    os.makedirs(target_image_dir)

# 遍历图像文件夹
for idx, row in image_list.iterrows():
    # 获取原图像的序号，例如00001.png, 00002.png...
    original_image_name = f'{idx+1:05d}.png'  # 假设图像文件按顺序命名 00001.png, 00002.png...
    original_image_path = os.path.join(source_image_dir, original_image_name)

    # 获取 image_list.txt 中对应的 orig_file 列的文件名
    new_image_name = row['orig_file'].replace('.jpg', '.png')  # 替换扩展名为 .png
    new_image_path = os.path.join(target_image_dir, new_image_name)

    # 检查文件是否存在，然后复制并重命名到目标目录
    if os.path.exists(original_image_path):
        shutil.copy2(original_image_path, new_image_path)
        print(f"Copied and renamed: {original_image_name} -> {new_image_name}")
    else:
        print(f"File not found: {original_image_name}")
