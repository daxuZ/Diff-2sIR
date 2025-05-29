import os
import cv2
from PIL import Image


# # 定义输入目录和输出目录
# input_dir = "/media/daxu/diskd/Datasets/CelebA-HQ/celeba-hq/masks_val_256_small_eval_rev"  # 替换为存放256分辨率掩码的目录路径
# output_dir = "/media/daxu/diskd/Datasets/CelebA-HQ/celeba-hq/masks_val_256_small_eval_rev_512"  # 替换为存放512分辨率掩码的目标目录路径
#
# # 确保输出目录存在
# os.makedirs(output_dir, exist_ok=True)
#
# # 遍历输入目录中的所有图像
# for filename in os.listdir(input_dir):
#     # 检查文件是否为图像文件
#     if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
#         input_path = os.path.join(input_dir, filename)
#         output_path = os.path.join(output_dir, filename)
#
#         # 打开图像并调整大小
#         with Image.open(input_path) as img:
#             # 确保掩码是单通道模式（黑白）
#             img = img.convert("L")
#             # 调整大小到 512x512
#             resized_img = img.resize((512, 512), Image.NEAREST)  # 使用最近邻插值保持掩码的黑白边缘
#             # 保存到目标目录
#             resized_img.save(output_path, "PNG")
#             print(f"已处理: {input_path} -> {output_path}")
#
# print("所有掩码已调整为512分辨率！")


def pixel_repeat(mask_256, scale=2):
    """
    通过像素重复将 256 分辨率的掩码放大到 512 分辨率。
    """
    return mask_256.repeat(scale, axis=0).repeat(scale, axis=1)


def generate_512_by_pixel_repeat(input_dir, output_dir):
    """
    使用像素重复方式生成 512 分辨率掩码。

    Args:
        input_dir (str): 256 掩码的目录。
        output_dir (str): 保存 512 掩码的目录。
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for file_name in os.listdir(input_dir):
        file_path = os.path.join(input_dir, file_name)
        mask_256 = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if mask_256 is None:
            continue

        # 像素重复放大
        mask_512 = pixel_repeat(mask_256, scale=2)

        # 保存结果
        output_path = os.path.join(output_dir, file_name)
        cv2.imwrite(output_path, mask_512)
        print(f"Saved: {output_path}")


# 输入和输出目录
input_dir = "/media/daxu/diskd/Datasets/CelebA-HQ/celeba-hq/masks_val_256_small_eval_rev"
output_dir = "/media/daxu/diskd/Datasets/CelebA-HQ/celeba-hq/masks_256_small_rev_2_512"

# 生成 512 分辨率掩码
generate_512_by_pixel_repeat(input_dir, output_dir)

