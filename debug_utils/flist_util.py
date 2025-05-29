# 读取.flist文件内容
with open('E:\Pycharm\Palette_Windows\datasets\celebahq_mat\lama_fourier_out_all.flist', 'r') as file:
    lines = file.readlines()

# 替换路径中的部分字符串
new_lines = [line.replace('/media/daxu/diskd/Datasets/CelebA-HQ/celeba-hq/lama_fourier_out/', 'D:\\Datasets\\CelebA-HQ\\celeba-hq\\mat_out_large\\') for line in lines]

# 将修改后的内容写回文件
with open('E:\Pycharm\Palette_Windows\datasets\paper_use\mat_large_windows.flist', 'w') as file:
    file.writelines(new_lines)

print("Paths updated successfully!")
