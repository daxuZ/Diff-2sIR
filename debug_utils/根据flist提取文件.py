import os
import shutil

# 定义 .flist 文件路径和目标目录
flist_path = "/media/daxu/diskd/Users/Administrator/Pycharm/Palette-Image-to-Image-Diffusion-Model/datasets/celebahq_mat/val_celebahq512_png.flist"  # 替换为你的 .flist 文件路径
target_dir = "/media/daxu/diskd/Datasets/CelebA-HQ/celeba-hq/images_val_png_512"  # 替换为目标目录路径

# 确保目标目录存在
os.makedirs(target_dir, exist_ok=True)

# 读取 .flist 文件
with open(flist_path, "r") as f:
    image_paths = f.readlines()

# 遍历每一张图像路径
for image_path in image_paths:
    image_path = image_path.strip()  # 去掉行末的换行符和多余空格
    if not os.path.isfile(image_path):
        print(f"图像文件未找到: {image_path}")
        continue

    # 获取图像文件名并构造目标路径
    image_name = os.path.basename(image_path)
    target_path = os.path.join(target_dir, image_name)

    # 将图像复制到目标目录
    shutil.copy(image_path, target_path)
    print(f"已复制: {image_path} -> {target_path}")

print("所有图像已提取完成！")
