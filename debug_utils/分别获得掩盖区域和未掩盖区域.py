import os
from PIL import Image
import numpy as np

# 设置修复图像和掩码图像的目录
restored_dir = '/media/daxu/diskd/Datasets/CelebA-HQ/celeba-hq/images_val_png'  # 修复图像目录
mask_dir = '/media/daxu/diskd/Datasets/CelebA-HQ/celeba-hq/masks_val_256_small_eval'  # 掩码图像目录

# 创建两个新的目录来保存掩盖区域和未掩盖区域的图像
unmasked_dir = '/media/daxu/diskd/Datasets/CelebA-HQ/celeba-hq/images_val_png_masked'  # 未掩盖区域的保存目录
masked_dir = '/media/daxu/diskd/Datasets/CelebA-HQ/celeba-hq/images_val_png_unmasked'  # 掩盖区域的保存目录

os.makedirs(masked_dir, exist_ok=True)
os.makedirs(unmasked_dir, exist_ok=True)

# 获取所有修复图像和掩码图像的文件名
restored_images = sorted(os.listdir(restored_dir))
mask_images = sorted(os.listdir(mask_dir))

# 遍历每一对修复图像和掩码图像
for restored_img_name, mask_img_name in zip(restored_images, mask_images):
    # 构造修复图像和掩码图像的路径
    restored_img_path = os.path.join(restored_dir, restored_img_name)
    mask_img_path = os.path.join(mask_dir, mask_img_name)

    # 打开修复图像和掩码图像
    restored_img = Image.open(restored_img_path).convert('RGB')
    mask_img = Image.open(mask_img_path).convert('L')  # L 模式是灰度模式

    # 确保图像大小一致
    restored_img = restored_img.resize((256, 256))
    mask_img = mask_img.resize((256, 256))

    # 将图像和掩码转换为 numpy 数组
    restored_array = np.array(restored_img)
    mask_array = np.array(mask_img)

    # 创建掩盖区域和未掩盖区域的图像
    # 掩盖区域：mask值为255的地方保留
    masked_array = restored_array * (mask_array[:, :, None] == 255)
    # 未掩盖区域：mask值为0的地方保留
    unmasked_array = restored_array * (mask_array[:, :, None] == 0)

    # 将 numpy 数组转换回图像
    masked_img = Image.fromarray(masked_array.astype(np.uint8))
    unmasked_img = Image.fromarray(unmasked_array.astype(np.uint8))

    # 保存图像
    masked_img.save(os.path.join(masked_dir, restored_img_name))
    unmasked_img.save(os.path.join(unmasked_dir, restored_img_name))

print("图像处理完成，已保存掩盖区域和未掩盖区域的图像。")
