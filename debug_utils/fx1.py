import os

# 定义图像存储目录
image_dir = '/media/daxu/diskd/Datasets/CelebA-HQ/celeba-hq/celeba_256'

# 读取训练集、验证集和测试集文件
def read_file(file_path):
    with open(file_path, 'r') as f:
        return [line.strip() for line in f]

train_files = read_file('train_set.txt')
val_files = read_file('val_set.txt')
test_files = read_file('test_set.txt')

# 生成图像路径
def generate_image_paths(file_list, image_dir):
    return [os.path.join(image_dir, file_name) for file_name in file_list]

train_paths = generate_image_paths(train_files, image_dir)
val_paths = generate_image_paths(val_files, image_dir)
test_paths = generate_image_paths(test_files, image_dir)

# 保存路径到 .flist 文件
def save_to_flist(file_paths, output_path):
    with open(output_path, 'w') as f:
        for path in file_paths:
            f.write(f"{path}\n")

save_to_flist(train_paths, 'train_celebahq.flist')
save_to_flist(val_paths, 'val_celebahq.flist')
save_to_flist(test_paths, 'test_celebahq.flist')

print("FLIST files have been created successfully.")
