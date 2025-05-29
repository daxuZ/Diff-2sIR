import os
import random
import shutil

# 定义目录路径
out_img_dir = '/media/daxu/diskd/下载/test_celebahq_unet500_mamba350_centermask/results/test/out_img'
gt_img_dir = '/media/daxu/diskd/下载/test_celebahq_unet500_mamba350_centermask/results/test/gt_img'
target_out_dir = '/media/daxu/diskd/下载/test_celebahq_unet500_mamba350_centermask/results/test/out_img8'
target_gt_dir = '/media/daxu/diskd/下载/test_celebahq_unet500_mamba350_centermask/results/test/gt_img8'

# 确保目标目录存在
os.makedirs(target_out_dir, exist_ok=True)
os.makedirs(target_gt_dir, exist_ok=True)

# 获取out_img目录下的所有图像文件名
out_img_files = [f for f in os.listdir(out_img_dir) if f.startswith('Out_') and f.endswith('.jpg')]

# 随机抽取2993张图像文件
sampled_out_img_files = random.sample(out_img_files, 1000)

# 复制out_img文件到目标目录
for file in sampled_out_img_files:
    src_file = os.path.join(out_img_dir, file)
    dst_file = os.path.join(target_out_dir, file)
    shutil.copy(src_file, dst_file)

# 复制对应的gt_img文件到另一个目标目录
for file in sampled_out_img_files:
    # 替换文件名前缀
    gt_file = file.replace('Out_', 'GT_')
    src_gt_file = os.path.join(gt_img_dir, gt_file)
    dst_gt_file = os.path.join(target_gt_dir, gt_file)
    # 复制对应的gt_img文件到目标目录
    if os.path.exists(src_gt_file):
        shutil.copy(src_gt_file, dst_gt_file)
    else:
        print(f"对应的gt_img文件 {gt_file} 不存在")

print("文件复制完成")
