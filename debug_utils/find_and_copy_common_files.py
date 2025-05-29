import os
import shutil


def find_and_copy_common_files(dir1, dir2, target_dir):
    # 获取目录1和目录2中的文件名
    files_dir1 = set(os.listdir(dir1))
    files_dir2 = set(os.listdir(dir2))

    # 找出相同的文件名
    common_files = files_dir1.intersection(files_dir2)

    # 创建目标目录（如果不存在）
    os.makedirs(target_dir, exist_ok=True)

    # 复制相同的文件到目标目录
    for file_name in common_files:
        src_path = os.path.join(dir1, file_name)
        if os.path.isfile(src_path):  # 确保这是一个文件而不是目录
            shutil.copy(src_path, target_dir)
            print(f'Copied {file_name} from {dir1} to {target_dir}')

        src_path = os.path.join(dir2, file_name)
        if os.path.isfile(src_path):  # 确保这是一个文件而不是目录
            shutil.copy(src_path, target_dir)
            print(f'Copied {file_name} from {dir2} to {target_dir}')
    print("Done!")


# 示例用法
dir1 = '/media/daxu/diskd/Users/Administrator/Pycharm/Palette-Image-to-Image-Diffusion-Model/experiments/test_inpainting_celeba_240718_142821/results/test/out_img'  # 替换为第一个目录的路径
dir2 = '/media/daxu/diskd/Users/Administrator/Pycharm/Palette-Image-to-Image-Diffusion-Model/experiments/mamba_skip_celeba256_500/results/test/out_img'  # 替换为第二个目录的路径
target_dir = '/experiments/test_inpainting_celeba_240718_142821/results/test/out1'  # 替换为目标目录的路径

find_and_copy_common_files(dir1, dir2, target_dir)
