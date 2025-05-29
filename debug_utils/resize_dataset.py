import os
import cv2

# 原始图像目录和输出目录
input_folder = "/media/daxu/diskd/Datasets/CelebA-HQ/celebahq1024"  # 包含 JPG 图像的目录
output_folder = "/media/daxu/diskd/Datasets/CelebA-HQ/celebahq256"  # 保存 PNG 图像的目录

# 创建输出目录（如果不存在）
os.makedirs(output_folder, exist_ok=True)

# 遍历原始目录中的所有 JPG 图像
for filename in os.listdir(input_folder):
    if filename.lower().endswith('.jpg'):
        # 构建文件的完整路径
        input_filepath = os.path.join(input_folder, filename)

        # 读取图像
        image = cv2.imread(input_filepath)

        # 调整图像分辨率为 256x256
        resized_image = cv2.resize(image, (256, 256))

        # 生成新的文件名，替换为 PNG 格式
        new_filename = filename.replace('.jpg', '.png')
        output_filepath = os.path.join(output_folder, new_filename)

        # 保存为 PNG 格式
        cv2.imwrite(output_filepath, resized_image)

        print(f"Converted: {filename} to {new_filename} and resized to 256x256.")
    elif filename.lower().endswith('.png'):
        input_filepath = os.path.join(input_folder, filename)
        image = cv2.imread(input_filepath)
        resized_image = cv2.resize(image, (256, 256))
        output_filepath = os.path.join(output_folder, filename)
        cv2.imwrite(output_filepath, resized_image)
        print(f"Converted: {filename} resized to 256x256.")


