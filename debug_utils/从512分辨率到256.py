import os
from PIL import Image

# 定义输入和输出目录
input_dir = "/media/daxu/diskd/Users/Administrator/Pycharm/AOT-GAN-for-Inpainting/output/comp_512_1"  # 替换为512分辨率图像的目录路径
output_dir = "/media/daxu/diskd/Users/Administrator/Pycharm/AOT-GAN-for-Inpainting/output/comp_512_256"  # 替换为保存256分辨率图像的目录路径

# 创建输出目录（如果不存在）
os.makedirs(output_dir, exist_ok=True)

# 遍历输入目录中的所有文件
for filename in os.listdir(input_dir):
    input_path = os.path.join(input_dir, filename)

    # 检查是否是图像文件（可以根据实际情况添加更多格式）
    if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
        # 打开图像
        with Image.open(input_path) as img:
            # 确保图像是 RGB 模式
            img = img.convert("RGB")

            # 调整图像大小到 256x256 分辨率
            img_resized = img.resize((256, 256), Image.LANCZOS)

            # 保存到输出目录
            output_path = os.path.join(output_dir, filename)
            img_resized.save(output_path)

print("图像已全部调整为 256 分辨率并保存到输出目录。")
