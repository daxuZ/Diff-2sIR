import os
import shutil


def copy_matching_images(dir_a, dir_b, dir_c):
    # 确保目标目录存在
    os.makedirs(dir_c, exist_ok=True)

    # 遍历目录 a 中的所有文件
    for filename in os.listdir(dir_a):
        # 构造目录 a 和 b 中的文件路径
        file_a = os.path.join(dir_a, filename)
        file_b = os.path.join(dir_b, filename)

        # 检查文件是否存在于目录 b
        if os.path.isfile(file_a) and os.path.isfile(file_b):
            # 复制文件到目录 c
            shutil.copy(file_a, dir_c)
            print(f"Copied: {filename}")


# 示例用法
# /media/daxu/diskd/Users/Administrator/Pycharm/MAT/test_sets/CelebA-HQ/images_val_png
dir_a = '/media/daxu/diskd/Users/Administrator/Pycharm/MAT/test_sets/CelebA-HQ/images_val_png'  # 替换为要抽取的目录路径
dir_b = '/experiments/415_150/results/test/out'  # 替换为目录 b 的路径
dir_c = '/experiments/415_150/results/test/gt'  # 替换为目录 c 的路径

copy_matching_images(dir_a, dir_b, dir_c)


# def copy_matching_images(dir_a, dir_b, dir_c):
#     # 确保目标目录存在
#     os.makedirs(dir_c, exist_ok=True)
#
#     # 遍历目录 a 中的所有文件
#     for filename in os.listdir(dir_a):
#         if filename.startswith("GT_") and filename.endswith(".jpg"):
#             # 去掉前缀 "GT_" 和后缀 ".jpg"
#             base_name = filename[3:-4]
#
#             # 构造目录 b 中的匹配文件名
#             matching_file_in_b = f"Out_{base_name}.jpg"
#
#             # 检查目录 b 中是否存在对应的文件
#             file_b = os.path.join(dir_b, matching_file_in_b)
#             if os.path.isfile(file_b):
#                 # 构造目录 a 和目标目录 c 中的文件路径
#                 file_a = os.path.join(dir_a, filename)
#                 shutil.copy(file_a, dir_c)
#                 print(f"Copied: {filename}")
#
#
# # 示例用法
# dir_a = '/media/daxu/diskd/下载/test_双重约束/test_inpainting_celebahq_240623_211320/results/test/out'  # 替换为目录 a 的路径
# dir_b = '/media/daxu/diskd/下载/test_mamba_seed0/results/test/test'  # 替换为目录 b 的路径
# dir_c = '/media/daxu/diskd/下载/test_双重约束/test_inpainting_celebahq_240623_211320/results/test/gt1'  # 替换为目录 c 的路径
#
# copy_matching_images(dir_a, dir_b, dir_c)

