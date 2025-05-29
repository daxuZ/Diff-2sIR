import os
import shutil  # 用于复制文件


def rename_and_save_images(src_directory, dest_directory):
    # 如果目标目录不存在，则创建
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)

    # 获取源目录中所有的图像文件，并按照文件名排序
    files = sorted([f for f in os.listdir(src_directory) if f.endswith('.png')])

    # 遍历所有文件并重命名保存到目标目录
    for index, file in enumerate(files, start=1):
        old_path = os.path.join(src_directory, file)  # 原文件路径
        new_name = f"image{index}_mask001.png"  # 新文件名
        new_path = os.path.join(dest_directory, new_name)  # 新文件路径

        # 复制文件到目标目录并重命名
        shutil.copy(old_path, new_path)
        print(f"Copied and renamed: {file} -> {new_name}")


# 设置源目录和目标目录
src_directory = "/media/daxu/diskd/Datasets/CelebA-HQ/celeba-hq/masks_val_256_large_eval_rev"  # 替换为你的源目录路径
dest_directory = "/media/daxu/diskd/Datasets/CelebA-HQ/celeba-hq/dirrirs2_test_large"  # 替换为你的目标目录路径

# 调用函数
rename_and_save_images(src_directory, dest_directory)
