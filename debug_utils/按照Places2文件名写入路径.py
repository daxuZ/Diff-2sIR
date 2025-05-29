import os
import re

# 设置你的路径
image_dir = r"D:\Datasets\Places2\debug_test"
output_flist = r"E:\Pycharm\Palette_Windows\datasets\places2\places2_test666.flist"

# 正则匹配 Places365_val_0_xxx.jpg 格式的文件
pattern = re.compile(r"Places365_val_(\d+)\.jpg")

# 获取所有符合命名格式的文件
image_files = []
try:
    for filename in os.listdir(image_dir):
        match = pattern.match(filename)
        if match:
            number = int(match.group(1))  # 提取数字部分
            full_path = os.path.normpath(os.path.join(image_dir, filename))  # 生成标准路径
            image_files.append((number, full_path))
except FileNotFoundError:
    print(f"错误: 目录 {image_dir} 不存在！")
    exit()

# 按编号排序
image_files = sorted(image_files, key=lambda x: x[0])

# 检查是否找到文件
if not image_files:
    print("没有找到符合命名格式的文件，请检查路径或文件名格式！")
    exit()

# 调试输出前 5 个匹配的文件
print("匹配的文件示例:", image_files[:5])

# 写入 .flist 文件
try:
    with open(output_flist, "w") as f:
        for _, filepath in image_files:
            f.write(filepath + "\n")
    print(f"已生成 {output_flist}，共 {len(image_files)} 个文件")
except IOError:
    print(f"错误: 无法写入文件 {output_flist}！请检查路径是否可写。")
