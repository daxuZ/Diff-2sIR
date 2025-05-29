from PIL import Image
import os

def convert_images(src_dir, dest_dir):
    # 创建目标目录（如果不存在的话）
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    # 遍历源目录中的所有文件
    for filename in os.listdir(src_dir):
        if filename.lower().endswith('.jpg'):
            # 构建完整的文件路径
            img_path = os.path.join(src_dir, filename)
            # 打开图像
            with Image.open(img_path) as img:
                # 创建新的文件名
                new_filename = os.path.splitext(filename)[0] + '.png'
                new_img_path = os.path.join(dest_dir, new_filename)
                # 保存为 PNG 格式
                img.save(new_img_path, 'PNG')

    print("转换完成!")

# 使用示例
source_directory = '/media/daxu/diskd/Datasets/CelebA-HQ/celeba-hq/celeba_512_mat'  # 源目录
destination_directory = '/media/daxu/diskd/Datasets/CelebA-HQ/celeba-hq/celebahq512_png'  # 目标目录

convert_images(source_directory, destination_directory)
