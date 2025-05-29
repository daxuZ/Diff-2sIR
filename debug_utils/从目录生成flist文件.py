import os


def save_image_paths_to_flist(directory, output_flist):
    """
    将指定目录中的图像路径按照文件名排序写入到 .flist 文件中。

    :param directory: 图像文件所在的目录
    :param output_flist: 输出的 .flist 文件路径
    """
    # 获取目录中的所有文件，并按文件名排序
    file_list = sorted(os.listdir(directory))  # 按文件名排序
    file_paths = [os.path.join(directory, file) for file in file_list if os.path.isfile(os.path.join(directory, file))]

    # 将文件路径写入到 .flist 文件中
    with open(output_flist, "w") as f:
        f.write("\n".join(file_paths))

    print(f"图像路径已写入到 {output_flist} 中，共 {len(file_paths)} 条。")


# 使用示例
input_directory = "/media/daxu/diskd/Datasets/CelebA-HQ/celeba-hq/masks_val_256_small_eval"  # 替换为你的图像目录路径
output_flist_path = "/media/daxu/diskd/Users/Administrator/Pycharm/Palette-Image-to-Image-Diffusion-Model/datasets/celebahq_mat/masks_val_256_small_eval.flist"  # 替换为你想保存的 .flist 文件路径

save_image_paths_to_flist(input_directory, output_flist_path)
