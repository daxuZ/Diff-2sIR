import pandas as pd
from PIL import Image
import io
import os

# 列出所有 .parquet 文件路径
parquet_files = [
    '/media/daxu/diskd/下载/train-00000-of-00006.parquet',
    '/media/daxu/diskd/下载/train-00001-of-00006.parquet',
    '/media/daxu/diskd/下载/train-00002-of-00006.parquet',
    '/media/daxu/diskd/下载/train-00003-of-00006.parquet',
    '/media/daxu/diskd/下载/train-00004-of-00006.parquet',
    '/media/daxu/diskd/下载/train-00005-of-00006.parquet',
    '/media/daxu/diskd/下载/validation-00000-of-00001.parquet'
]

# 创建保存 PNG 文件的目录
output_dir = '/media/daxu/diskd/Datasets/CelebA_HQ'
os.makedirs(output_dir, exist_ok=True)

# 从编号1开始
image_idx = 1

# 遍历所有 .parquet 文件
for parquet_file in parquet_files:
    # 读取 parquet 文件
    df = pd.read_parquet(parquet_file)

    # 遍历每个文件中的 DataFrame 并将图像数据保存为 PNG 格式
    for _, row in df.iterrows():
        # 提取字典中的 'bytes' 部分（图像的二进制数据）
        image_data = row['image']['bytes']

        # 将二进制数据转换为 PIL 图像对象
        image = Image.open(io.BytesIO(image_data))

        # 使用官方格式的文件名保存，例如 000001.png
        official_filename = f'{image_idx:06d}.png'  # 保持编号从1开始

        # 保存为 PNG 文件
        image.save(os.path.join(output_dir, official_filename))

        # 增加图片编号
        image_idx += 1

print("所有 PNG 图像保存完成！")
