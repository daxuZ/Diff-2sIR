import os
import shutil

def rename_images_by_order(flist_path, src_dir, dst_dir, total_images):
    """
    按固定顺序将源文件名重命名为 flist 文件中的目标文件名，并保存到目标目录。

    :param flist_path: flist 文件路径，包含目标文件名
    :param src_dir: 源图像目录
    :param dst_dir: 目标图像目录
    :param total_images: 源文件的总数（如 2993）
    """
    # 创建目标目录（如果不存在）
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    # 读取 flist 文件内容并提取目标文件名
    with open(flist_path, 'r') as f:
        target_filenames = [os.path.basename(line.strip()) for line in f.readlines()]

    # 检查 flist 文件的数量是否匹配总图像数量
    if len(target_filenames) != total_images:
        print("flist 文件中的目标文件名数量与总图像数量不匹配！")
        return

    # 按顺序生成源文件名列表
    # src_filenames = [f"image{i}_mask001.png" for i in range(1, total_images+1)]  # (1, total_images + 1)
    # 处理 M2S 的结果使用此行代码
    src_filenames = [f"{i:04d}.png" for i in range(0, total_images)]

    # 按顺序重命名并保存
    for src_name, target_name in zip(src_filenames, target_filenames):
        src_path = os.path.join(src_dir, src_name)
        dst_path = os.path.join(dst_dir, target_name)

        # 检查源文件是否存在
        if os.path.exists(src_path):
            shutil.copy(src_path, dst_path)  # 复制并重命名
            print(f"已将 {src_name} 重命名为 {target_name} 并保存到 {dst_dir}")
        else:
            print(f"源文件 {src_name} 不存在！")

# 使用示例
flist_path = "/media/daxu/diskd/Users/Administrator/Pycharm/Palette-Image-to-Image-Diffusion-Model/datasets/celebahq_mat/mat_out_png.flist"  # 替换为 flist 文件路径
src_dir = "/media/daxu/diskd/Users/Administrator/Pycharm/M2S/output_256_large/outImg"  # 替换为源图像目录路径
dst_dir = "/media/daxu/diskd/Datasets/CelebA-HQ/celeba-hq/M2S_out_large"  # 替换为目标目录路径
total_images = 2993  # 图像总数，例如 2993

rename_images_by_order(flist_path, src_dir, dst_dir, total_images)
