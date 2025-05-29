# 读取 a.flist 和 b.flist 文件内容
with open('/media/daxu/diskd/Users/Administrator/Pycharm/Palette-Image-to-Image-Diffusion-Model/test.flist', 'r') as file_a, open('/media/daxu/diskd/Users/Administrator/Pycharm/Palette-Image-to-Image-Diffusion-Model/debug_utils/test_celebahq.flist', 'r') as file_b:
    a_lines = file_a.readlines()
    b_lines = file_b.readlines()

# 提取文件名（去掉路径，只保留文件名）
a_files = {line.strip().split('/')[-1]: line.strip() for line in a_lines}
b_files = {line.strip().split('/')[-1]: line.strip() for line in b_lines}

# 找出重复的文件名
common_files = set(a_files.keys()).intersection(b_files.keys())

# 将 a.flist 中重复的行写入 c.flist
with open('/media/daxu/diskd/Users/Administrator/Pycharm/Palette-Image-to-Image-Diffusion-Model/debug_utils/datasets/mnist/celebahq_test_both', 'w') as file_c:
    for filename in common_files:
        file_c.write(a_files[filename] + '\n')

print(f"找到 {len(common_files)} 个重复文件，并保存到 c.flist 中。")
