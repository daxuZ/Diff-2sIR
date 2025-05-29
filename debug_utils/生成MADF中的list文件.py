# 定义原始图像和掩码图像路径的 .flist 文件
original_flist_path = '/media/daxu/diskd/Users/Administrator/Pycharm/Palette-Image-to-Image-Diffusion-Model/datasets/celebahq_mat/val_celebahq_png.flist'  # 替换为原始图像 .flist 文件路径
mask_flist_path = '/media/daxu/diskd/Users/Administrator/Pycharm/Palette-Image-to-Image-Diffusion-Model/datasets/celebahq_mat/masks_val_256_small_eval.flist'  # 替换为掩码图像 .flist 文件路径
output_flist_path = '/media/daxu/diskd/Users/Administrator/Pycharm/Palette-Image-to-Image-Diffusion-Model/datasets/celebahq_mat/MADF.flist'  # 输出的 .flist 文件路径

# 读取两个 .flist 文件内容
with open(original_flist_path, 'r') as original_flist_file:
    original_lines = original_flist_file.readlines()

with open(mask_flist_path, 'r') as mask_flist_file:
    mask_lines = mask_flist_file.readlines()

# 确保两个文件的行数一致
if len(original_lines) != len(mask_lines):
    raise ValueError("原始图像文件和掩码文件的行数不一致！请检查输入文件。")

# 合并路径，并保存到输出文件
with open(output_flist_path, 'w') as output_file:
    for original_path, mask_path in zip(original_lines, mask_lines):
        # 去除每行末尾的换行符
        original_path = original_path.strip()
        mask_path = mask_path.strip()
        # 将原始图像路径和掩码图像路径合成一行，并使用四个空格间隔
        combined_line = f"{original_path}    {mask_path}\n"
        output_file.write(combined_line)

print(f"合成完成！结果已保存到：{output_flist_path}")
