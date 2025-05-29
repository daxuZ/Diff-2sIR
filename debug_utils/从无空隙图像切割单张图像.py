import os
import cv2
import numpy as np
from PIL import Image

# 读取原始的大图
input_image_path = "E:\\visio_img\\pictures\\x+h_debug_unet_paper1\\image_20250327_205205.png"
img = cv2.imread(input_image_path)  # 读取图片（BGR 格式）
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换为 RGB

# 设定行列数
rows, cols = 2, 4  # 2 行 × 4 列
h, w, _ = img.shape  # 获取大图的尺寸
cell_h, cell_w = h // rows, w // cols  # 计算单张图像的尺寸

# 创建保存目录
output_dir = "E:\\visio_img\\pictures\\unet_切割单张图像\\3"
os.makedirs(output_dir, exist_ok=True)

# 遍历分割图像并保存
for r in range(rows):
    for c in range(cols):
        x1, y1 = c * cell_w, r * cell_h  # 计算切割区域
        x2, y2 = x1 + cell_w, y1 + cell_h
        cropped_img = img[y1:y2, x1:x2]  # 进行切割

        # 保存图像
        output_path = os.path.join(output_dir, f"sub_image_{r}_{c}.png")
        Image.fromarray(cropped_img).save(output_path)

print(f"已成功分割并保存 {rows * cols} 张子图到 {output_dir}")
