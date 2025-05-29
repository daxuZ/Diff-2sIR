import os
import random

# 定义目录路径和保存文件路径
image_dir = '/media/daxu/diskd/Datasets/CelebA-HQ/celeba-hq/celeba_256'
flist_save_path = '../flist'

# 获取目录中所有图像文件的文件名列表
image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.jpg') or f.endswith('.png')]

# 随机打乱图像文件列表
random.shuffle(image_files)

# 计算训练集和测试集的图像数量
num_train = 24000
num_test = len(image_files) - num_train

# 创建保存文件列表的文件夹
os.makedirs(flist_save_path, exist_ok=True)

# 生成训练集文件列表
with open(os.path.join(flist_save_path, 'celebahq/celebahq_train.flist'), 'w') as f:
    for image_file in image_files[:num_train]:
        print(image_file, file=f)

# 生成测试集文件列表
with open(os.path.join(flist_save_path, 'celebahq/celebahq_test.flist'), 'w') as f:
    for image_file in image_files[num_train:]:
        print(image_file, file=f)

# 读取测试集文件列表
test_flist_path = os.path.join(flist_save_path, 'celebahq/celebahq_test.flist')
with open(test_flist_path, 'r') as f:
    test_image_files = f.readlines()
test_image_files = [line.strip() for line in test_image_files]

# 随机抽取 50 张图像作为验证集
val_image_files = random.sample(test_image_files, 50)

# 生成验证集文件列表
with open(os.path.join(flist_save_path, 'celebahq/celebahq_val.flist'), 'w') as f:
    for image_file in val_image_files:
        print(image_file, file=f)
