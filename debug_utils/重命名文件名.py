import os

# 设置你的目录路径
directory = '/media/daxu/diskd/Users/Administrator/Pycharm/Palette-Image-to-Image-Diffusion-Model/experiments/lama_fourier_debug_1500_300/results/test/out_img'

for filename in os.listdir(directory):
    if filename.startswith('GT_'):
        new_name = filename[3:]  # 去掉 'GT_' 前缀
        os.rename(os.path.join(directory, filename), os.path.join(directory, new_name))
    elif filename.startswith('Out_'):
        new_name = filename[4:]  # 去掉 'Out_' 前缀
        os.rename(os.path.join(directory, filename), os.path.join(directory, new_name))

print("重命名完成！")

